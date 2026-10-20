"""Independent endpoint bookkeeping/checker. STOI/PESQ reuse locked official implementations,
not an independent implementation of either published metric. No model imports."""
import math,random,warnings

def groups(cases):
    result={'all':cases}
    for n in ['N1','N2']:result['noise_'+n]=[r for r in cases if r['noise']==n]
    for q in [-18,0]:result['snr_'+str(q)]=[r for r in cases if r['snr_db']==q]
    for n in ['N1','N2']:
        for q in [-18,0]:result[n+'_'+str(q)]=[r for r in cases if r['noise']==n and r['snr_db']==q]
    for k,pre in [('speaker','speaker_'),('id','utterance_')]:
        for v in sorted({r[k] for r in cases}):result[pre+v]=[r for r in cases if r[k]==v]
    return result

def verify_headroom(m,reg,aggregate):
    scores={i['key']:i['score'] for i in m['items']};cases=reg['cases'];allgroups=groups(cases);h=m['headroom'];assert set(h['strata'])==set(allgroups)
    def counts(rows):
        a=sum(scores[r['basename']+'__frontend']['word']['edit_distance'] for r in rows);b=sum(scores[r['basename']+'__off']['word']['edit_distance'] for r in rows);N=sum(scores[r['basename']+'__frontend']['word']['N'] for r in rows)
        c=sum(min(scores[r['basename']+'__frontend']['word']['edit_distance'],scores[r['basename']+'__off']['word']['edit_distance']) for r in rows)
        return a,b,c,N
    for label,rows in allgroups.items():
        d=h['strata'][label];a,b,c,N=counts(rows);assert (d['E_frontend'],d['E_off'],d['E_oracle'],d['N'])==(a,b,c,N)
        assert d['E_best_fixed']==min(a,b) and d['best_fixed']==('frontend' if a<=b else 'off')
        assert d['front_errors_saved']==a-c and d['off_errors_saved']==b-c
        assert d['headroom']==((min(a,b)-c)/N if N else None) and d['off_minus_frontend']==((b-a)/N if N else None)
        chosen=[];nw=nb=nt=0
        for r,choice in zip(rows,d['choices']):
            f=scores[r['basename']+'__frontend'];o=scores[r['basename']+'__off'];ef=f['word']['edit_distance'];eo=o['word']['edit_distance'];arm='frontend' if ef<=eo else 'off'
            assert choice==dict(basename=r['basename'],arm=arm);chosen.append(f if arm=='frontend' else o);nw+=ef<eo;nb+=eo<ef;nt+=eo==ef
        assert d['frontend_wins']==nw and d['off_wins']==nb and d['ties']==nt and len(d['choices'])==len(rows)
        assert d['WER_oracle']==aggregate(chosen)
        for arm in ['noisy','frontend','off']:assert d['arms'][arm]==aggregate([scores[r['basename']+'__'+arm] for r in rows])
    rng=random.Random(20261019);speakers=sorted({r['speaker'] for r in cases});v=[[],[]]
    for i in range(10000):
        selected=[rng.choice(speakers) for _ in range(len(speakers))];rows=[]
        for speaker in selected:rows.extend(r for r in cases if r['speaker']==speaker)
        a,b,c,N=counts(rows)
        if N:v[0].append((b-a)/N);v[1].append((min(a,b)-c)/N)
    expected=[]
    for arr in v:
        arr.sort();ci=[]
        for q in [.025,.975]:
            if not arr:ci.append(None);continue
            index=(len(arr)-1)*q;lo=int(index);hi=math.ceil(index);ci.append(arr[lo]+(arr[hi]-arr[lo])*(index-lo))
        expected.append(ci)
    assert h['bootstrap']['off_minus_frontend']==expected[0] and h['bootstrap']['headroom']==expected[1]
    assert h['bootstrap']['seed']==20261019 and h['bootstrap']['replicates']==10000 and h['bootstrap']['clusters']==speakers
    return dict(strata=len(allgroups),bootstrap_replicates=10000)

def verify_quality(m,reg,INPUT,OUT):
    import numpy as np,soundfile as sf
    from pystoi import stoi
    from pesq import pesq
    rows=m['quality']['per_case'];assert len(rows)==32;native={r['basename']:r for r in m['native_cases']};clean={g['id']:g for g in reg['groups']};by={}
    def metric(fn,short=False):
        notes=[];error=None;raw=None
        try:
            with warnings.catch_warnings(record=True) as ws:
                warnings.simplefilter('always')
                try:raw=float(fn())
                finally:notes=[dict(category=w.category.__name__,message=str(w.message)) for w in ws]
            if not math.isfinite(raw):error='nonfinite metric'
            elif short and raw==1e-5 and any('Not enough STFT frames' in n['message'] for n in notes):error='STOI short-frame sentinel (raw preserved)'
        except Exception as e:error=repr(e)
        return dict(value=raw if error is None else None,raw_value=raw if raw is None or math.isfinite(raw) else None,raw_value_repr=repr(raw) if raw is not None and not math.isfinite(raw) else None,error=error,warnings=notes)
    def close(a,b):return a==b if a is None or b is None else math.isclose(a,b,rel_tol=1e-10,abs_tol=1e-9)
    for case,row in zip(reg['cases'],rows):
        assert row['basename']==case['basename'];by[case['basename']]=row
        ref,sr=sf.read(INPUT/clean[case['id']]['scaledclean']['wav'],dtype='float64');assert sr==16000
        n=native[case['basename']]
        for arm,path in [('noisy',INPUT/case['wav']),('frontend',OUT/n['frontend_pcm']['path']),('off',OUT/n['off_pcm']['path'])]:
            x,fs=sf.read(path,dtype='float64');assert fs==16000 and len(ref)==len(x)
            a=ref-ref.mean();b=x-x.mean();den=float(np.dot(a,a));val=None;reason=None
            if den<=0:reason='nonpositive reference energy'
            else:
                scale=float(np.dot(a,b))/den;signal=float(np.dot(scale*a,scale*a));res=b-scale*a;err=float(np.dot(res,res))
                if signal<=0:reason='zero projection'
                elif err<=0:reason='perfect residual'
                else:val=10*math.log10(signal/err)
            got=row['arms'][arm]['si_sdr'];assert close(got['value'],val) and got['error']==reason
            assert row['arms'][arm]['stoi']==metric(lambda:stoi(ref,x,16000,extended=False),True)
            assert row['arms'][arm]['pesq']==metric(lambda:pesq(16000,ref,x,mode='wb'))
        for key in ['si_sdr','stoi','pesq']:
            f=row['arms']['frontend'][key]['value'];b=row['arms']['off'][key]['value'];assert close(row['off_minus_frontend'][key],b-f if f is not None and b is not None else None)
    for label,cases in groups(reg['cases']).items():
        selected=[by[r['basename']] for r in cases]
        for key,summary in m['quality']['strata'][label].items():
            for arm in ['noisy','frontend','off']:
                vals=[r['arms'][arm][key]['value'] for r in selected if r['arms'][arm][key]['value'] is not None];assert summary[arm]['valid_n']==len(vals) and summary[arm]['missing_n']==len(selected)-len(vals);assert close(summary[arm]['mean'],math.fsum(vals)/len(vals) if vals else None)
            vals=[r['off_minus_frontend'][key] for r in selected if r['off_minus_frontend'][key] is not None];d=summary['off_minus_frontend_paired'];assert d['paired_n']==len(vals) and d['missing_pair_n']==len(selected)-len(vals);assert close(d['mean'],math.fsum(vals)/len(vals) if vals else None)
    return dict(cases=len(rows),arms=96,STOI_PESQ='recomputed same frozen official libraries, independent bookkeeping; not independent algorithms',SI_SDR='independent centered dot-product implementation')

def verify_endpoints(m,reg,aggregate,INPUT,OUT):
    return dict(headroom=verify_headroom(m,reg,aggregate),quality=verify_quality(m,reg,INPUT,OUT))
