"""One saved-JSON interpretation check; no model, audio or new success gate."""
from pathlib import Path
import hashlib, json, math, statistics
ROOT=Path(__file__).resolve().parents[1]
FILES={a:ROOT/f'experiments/p8_cpel_native_pair_ablation_20261020/metrics_{a}.json' for a in ('N','D')}
EXPECTED={'N':'eae9d47fcf6e5387428918a18d020a6b104a7dbbec97a30f0521fc783689b598','D':'1174f563fb610acfd27055ebec27d5135c89bf17065b4f6598a28511bba93fc1'}
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
pre={a:sha(p) for a,p in FILES.items()};assert pre==EXPECTED
saved={a:json.loads(p.read_text()) for a,p in FILES.items()}
assert all(x['status']=='completed' for x in saved.values())
means=lambda x:statistics.mean(x)
all_rows=[]; summaries={}
for kind in ('speech_tukey','speech_tukey_pcm16'):
    tables={};meta={}
    for arm,x in saved.items():
        rows=x['evaluations']['final_'+arm];assert len(rows)==64
        table={};meta[arm]={}
        for row in rows:
            controls=[c for c in row['controls'] if c['kind']==kind];assert len(controls)==1
            c=controls[0];assert c['status']=='evaluated' and len(c['parents'])==2
            meta[arm][row['utt']]=[c[k] for k in ('crop_start','normfactor','event_center','center','gaussian_seed')]
            for p in c['parents']:
                key=(row['utt'],p['parent']);assert key not in table
                assert p['valid'] and not p['zero_response']
                values={k:float(p[k]) for k in ('orth','response_energy','projection_gain')}
                assert all(math.isfinite(v) for v in values.values())
                assert values['response_energy']>0 and values['orth']>=0
                table[key]=values
        assert len(table)==128;tables[arm]=table
    assert tables['N'].keys()==tables['D'].keys() and meta['N']==meta['D']
    rows=[]; per_file={}
    for (utt,parent),n in tables['N'].items():
        d=tables['D'][(utt,parent)];zn=n['orth']/n['response_energy'];zd=d['orth']/d['response_energy']
        gn,gd=n['projection_gain'],d['projection_gain']
        sign='both_positive' if gn>0 and gd>0 else 'both_negative' if gn<0 and gd<0 else 'zero_or_sign_flip'
        r=dict(kind=kind,utt=utt,parent=parent,Z_N=zn,Z_D=zd,delta_Z=zd-zn,
               relative_Z_change=None if zn==0 else zd/zn-1,gain_N=gn,gain_D=gd,gain_sign_class=sign)
        rows.append(r);per_file.setdefault(utt,[]).append(r)
    assert len(per_file)==64 and all(len(v)==2 for v in per_file.values())
    av={a:means([means([r['Z_'+a] for r in rr]) for rr in per_file.values()]) for a in ('N','D')}
    sign_counts={s:sum(r['gain_sign_class']==s for r in rows) for s in ('both_positive','both_negative','zero_or_sign_flip')}
    rel=[r['relative_Z_change'] for r in rows if r['relative_Z_change'] is not None]
    # Exact signs and >1e-5 bins are descriptive float32-roundoff guards, not a scientific success criterion.
    pos=[r for r in rows if r['gain_sign_class']=='both_positive'];eps=1e-5
    summaries[kind]=dict(files=64,parents=128,undefined_Z=0,undefined_relative_Z=sum(r['relative_Z_change'] is None for r in rows),
        file_equal_mean_Z=av,relative_change_of_file_equal_means=av['D']/av['N']-1,
        parent_relative_Z_min=min(rel),parent_relative_Z_median=statistics.median(rel),parent_relative_Z_max=max(rel),gain_sign_counts=sign_counts,
        positive_gain_parent_relative_changes=dict(decrease_beyond_roundoff_guard=sum(r['relative_Z_change'] is not None and r['relative_Z_change'] < -eps for r in pos),
            increase_beyond_roundoff_guard=sum(r['relative_Z_change'] is not None and r['relative_Z_change'] > eps for r in pos),
            within_guard=sum(r['relative_Z_change'] is not None and abs(r['relative_Z_change'])<=eps for r in pos)),
        nonpositive_gain_rows=[r for r in rows if r['gain_sign_class']!='both_positive'])
    all_rows.extend(rows)
assert {a:sha(p) for a,p in FILES.items()}==pre
result=dict(kind='post_hoc_saved_scalar_interpretation_not_new_model_experiment',source_sha256=pre,source_unchanged=True,
    statistic='Z=orth/response_energy; common k=sigma^2/alpha^2 cancels in matched N/D comparison',
    aggregation='parent Z first, two parents averaged within file, then 64 files equally weighted',
    roundoff_guard_relative=1e-5,guard_is_not_success_gate=True,summary=summaries,rows=all_rows,
    limits=['Only P-weighted response geometry, not full-vector distance or semantic identity',
            'Z invariance is necessary, not sufficient for positive scalar contraction',
            'Negative gain must not be called positive direction alignment',
            'Same exploratory CV and precision sensitivity, not independent evidence',
            'No inference, PCM, weights, bootstrap, retraining or old gate change'])
out=ROOT/'research_memos/p8_cpel_shape_ratio_analysis_20261020.json';out.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
print(json.dumps(dict(summary=summaries,source_sha256=pre,bytes=out.stat().st_size),ensure_ascii=False,indent=2))
