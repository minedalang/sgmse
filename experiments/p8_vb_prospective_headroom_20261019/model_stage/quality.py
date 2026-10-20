"""Fixed PCM endpoints, no external project gain/lag fit. Metric internals disclosed."""
import math,warnings

def sisdr(reference,estimate):
    if not reference or len(reference)!=len(estimate):return None,'empty/length mismatch'
    sm=math.fsum(reference)/len(reference);xm=math.fsum(estimate)/len(estimate)
    s=[v-sm for v in reference];x=[v-xm for v in estimate];ss=math.fsum(v*v for v in s)
    if ss<=0:return None,'nonpositive reference energy'
    alpha=math.fsum(a*b for a,b in zip(x,s))/ss;signal=math.fsum((alpha*v)**2 for v in s);residual=math.fsum((a-alpha*b)**2 for a,b in zip(x,s))
    if signal<=0:return None,'zero projection'
    if residual<=0:return None,'perfect residual'
    return 10*math.log10(signal/residual),None

def metric(fn,short_stoi=False):
    notes=[];raw=None;error=None
    try:
        with warnings.catch_warnings(record=True) as ws:
            warnings.simplefilter('always')
            try:raw=float(fn())
            finally:notes=[dict(category=w.category.__name__,message=str(w.message)) for w in ws]
        if not math.isfinite(raw):error='nonfinite metric'
        elif short_stoi and raw==1e-5 and any('Not enough STFT frames' in n['message'] for n in notes):error='STOI short-frame sentinel (raw preserved)'
    except Exception as e:error=repr(e)
    return dict(value=raw if error is None else None,raw_value=raw if raw is None or math.isfinite(raw) else None,raw_value_repr=repr(raw) if raw is not None and not math.isfinite(raw) else None,error=error,warnings=notes)

def measure_all(registry,native,input_root,out_root,checkpoint):
    import numpy as np,soundfile as sf
    from pystoi import stoi
    from pesq import pesq
    from headroom import strata
    groups={r['id']:r for r in registry['groups']};byname={r['basename']:r for r in native};rows=[]
    def read(p):
        x,sr=sf.read(p,dtype='float64');assert sr==16000 and x.ndim==1;return x
    for case in registry['cases']:
        checkpoint();ref=read(input_root/groups[case['id']]['scaledclean']['wav']);n=byname[case['basename']];arms={}
        for arm,p in [('noisy',input_root/case['wav']),('frontend',out_root/n['frontend_pcm']['path']),('off',out_root/n['off_pcm']['path'])]:
            checkpoint();x=read(p);assert len(x)==len(ref),'Quality fixed length mismatch';v,e=sisdr(ref.tolist(),x.tolist())
            arms[arm]=dict(si_sdr=dict(value=v,raw_value=v,error=e,warnings=[]),stoi=metric(lambda:stoi(ref,x,16000,extended=False),True),pesq=metric(lambda:pesq(16000,ref,x,mode='wb')))
        rows.append(dict(basename=case['basename'],arms=arms,off_minus_frontend={k:(arms['off'][k]['value']-arms['frontend'][k]['value'] if arms['off'][k]['value'] is not None and arms['frontend'][k]['value'] is not None else None) for k in ['si_sdr','stoi','pesq']}))
    by={r['basename']:r for r in rows};summary={}
    for label,cases in strata(registry['cases']):
        s={};selected=[by[r['basename']] for r in cases]
        for key in ['si_sdr','stoi','pesq']:
            d={}
            for arm in ['noisy','frontend','off']:
                vals=[r['arms'][arm][key]['value'] for r in selected if r['arms'][arm][key]['value'] is not None];d[arm]=dict(mean=math.fsum(vals)/len(vals) if vals else None,valid_n=len(vals),missing_n=len(selected)-len(vals))
            vals=[r['off_minus_frontend'][key] for r in selected if r['off_minus_frontend'][key] is not None];d['off_minus_frontend_paired']=dict(mean=math.fsum(vals)/len(vals) if vals else None,paired_n=len(vals),missing_pair_n=len(selected)-len(vals));s[key]=d
        summary[label]=s
    return dict(per_case=rows,strata=summary,preprocessing='Actual PCM full length/no project gain-lag fitting. STOI internal10k resampling/silence removal; PESQ internaljointpeak scaling/intrinsic processing; not human naturalness or verified phoneme identity.')
