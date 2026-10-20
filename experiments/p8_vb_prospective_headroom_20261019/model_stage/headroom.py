"""Offline sentence candidate-pool diagnostic. No candidate editing/deployment selector."""
import random,math

def strata(cases):
    yield 'all',cases
    for n in ['N1','N2']:yield 'noise_'+n,[r for r in cases if r['noise']==n]
    for q in [-18,0]:yield 'snr_'+str(q),[r for r in cases if r['snr_db']==q]
    for n in ['N1','N2']:
        for q in [-18,0]:yield n+'_'+str(q),[r for r in cases if r['noise']==n and r['snr_db']==q]
    for s in sorted({r['speaker'] for r in cases}):yield 'speaker_'+s,[r for r in cases if r['speaker']==s]
    for s in sorted({r['id'] for r in cases}):yield 'utterance_'+s,[r for r in cases if r['id']==s]

def stats(rows,score):
    f=[score[r['basename']+'__frontend'] for r in rows];b=[score[r['basename']+'__off'] for r in rows]
    ef=sum(r['word']['edit_distance'] for r in f);eb=sum(r['word']['edit_distance'] for r in b);N=sum(r['word']['N'] for r in f)
    choices=['frontend' if a['word']['edit_distance']<=z['word']['edit_distance'] else 'off' for a,z in zip(f,b)]
    eo=sum(min(a['word']['edit_distance'],z['word']['edit_distance']) for a,z in zip(f,b))
    div=lambda v:v/N if N else None
    return dict(N=N,E_frontend=ef,E_off=eb,E_best_fixed=min(ef,eb),best_fixed='frontend' if ef<=eb else 'off',E_oracle=eo,headroom=div(min(ef,eb)-eo),off_minus_frontend=div(eb-ef),front_errors_saved=ef-eo,off_errors_saved=eb-eo,frontend_wins=sum(a['word']['edit_distance']<z['word']['edit_distance'] for a,z in zip(f,b)),off_wins=choices.count('off'),ties=sum(a['word']['edit_distance']==z['word']['edit_distance'] for a,z in zip(f,b)),choices=[dict(basename=r['basename'],arm=c) for r,c in zip(rows,choices)])

def percentile(values,q):
    x=sorted(v for v in values if v is not None)
    if not x:return None
    t=(len(x)-1)*q;i=math.floor(t);j=math.ceil(t);return x[i]+(x[j]-x[i])*(t-i)

def analyze(items,cases,aggregate):
    score={r['key']:r['score'] for r in items};out={}
    for name,rows in strata(cases):
        d=stats(rows,score);d['arms']={a:aggregate([score[r['basename']+'__'+a] for r in rows]) for a in ['noisy','frontend','off']}
        d['WER_oracle']=aggregate([score[c['basename']+'__'+c['arm']] for c in d['choices']]);out[name]=d
    speakers=sorted({r['speaker'] for r in cases});rng=random.Random(20261019);draws=[]
    for _ in range(10000):
        selected=[rng.choice(speakers) for _ in speakers];rows=[r for s in selected for r in cases if r['speaker']==s];d=stats(rows,score);draws.append([d['off_minus_frontend'],d['headroom']])
    return dict(strata=out,bootstrap=dict(seed=20261019,replicates=10000,clusters=speakers,percentile_definition='linear (n-1)*q, exploratory only; four clusters not reliable coverage',off_minus_frontend=[percentile([v[0] for v in draws],q) for q in [.025,.975]],headroom=[percentile([v[1] for v in draws],q) for q in [.025,.975]]))
