"""I1b only: deterministic Clean-oracle candidate and in-memory operation arithmetic.
No labels of phoneme presence, classifiers, new waveform files or model calls."""
from pathlib import Path
import hashlib,io,json,math,os,sys
for key in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS']:os.environ[key]='1'
OUT=Path(__file__).resolve().parent

def choose(rows,key,maximize=True):
    defined=[r for r in rows if r[key] is not None and math.isfinite(r[key])]
    if not defined:return None
    optimum=(max if maximize else min)(r[key] for r in defined)
    return min((r for r in defined if abs(r[key]-optimum)<=1e-12),key=lambda r:r['start'])
def separated(a,b,gap=160):return a+1280+gap<=b or b+1280+gap<=a

def features(s):
    import numpy as np
    h=.5-.5*np.cos(2*np.pi*np.arange(320)/319);rows=[];previous=None
    for j,start in enumerate(range(0,len(s)-319,160)):
        frame=s[start:start+320];power=np.abs(np.fft.rfft(frame*h,n=512))**2;total=float(power.sum());q=power/total if total>0 and math.isfinite(total) else None
        R=float(q[64:241].sum()) if q is not None else None
        F=float(np.maximum(q[64:241]-previous[64:241],0).sum()) if q is not None and previous is not None else None
        rows.append(dict(frame=j,start=start,R=R,F=F,Q_reason=None if q is not None else 'nonpositive or nonfinite spectral power',F_reason=None if F is not None else 'first frame or undefined neighboring Q'))
        previous=q
    return rows

def select(s,rows):
    import numpy as np
    L=len(s);grid=list(range(3200,L-3200-1280+1,160));trans=[];sustain=[]
    for row in rows:
        start=160*row['frame']-480
        if start>=3200 and start+1280<=L-3200:trans.append(dict(start=start,end=start+1280,score=row['F'],eligible=row['F'] is not None and row['F']>0,reason=None if row['F'] is not None and row['F']>0 else 'no positive finite F'))
    t=choose([r for r in trans if r['eligible']],'score');tstart=t['start'] if t else None
    for start in grid:
        indices=[start//160+k for k in range(7)];values=[rows[i]['R'] for i in indices];defined=all(v is not None for v in values);score=sum(values)/7 if defined else None
        sep=tstart is None or separated(start,tstart);eligible=defined and score>0 and sep
        sustain.append(dict(start=start,end=start+1280,score=score,eligible=eligible,reason=None if eligible else 'undefined/nonpositive R or overlap transient'))
    u=choose([r for r in sustain if r['eligible']],'score');targets=[r['start'] for r in [t,u] if r is not None];events=[];controls=[]
    for kind,target in [('transient',t),('sustained',u)]:
        control_rows=[]
        if target is not None:
            energy=float(np.dot(s[target['start']:target['start']+1280],s[target['start']:target['start']+1280]))
            for start in grid:
                segment=s[start:start+1280];ce=float(np.dot(segment,segment));eligible=all(separated(start,v) for v in targets)
                control_rows.append(dict(event=kind,start=start,end=start+1280,energy=ce,target_energy=energy,absolute_energy_difference=abs(ce-energy),eligible=eligible,reason=None if eligible else 'overlap selected target plus gap'))
            ctrl=choose([r for r in control_rows if r['eligible']],'absolute_energy_difference',False)
        else:ctrl=None
        events.append(dict(event=kind,status='available' if target is not None else 'unavailable',reason=None if target is not None else 'no positive defined nonoverlapping candidate',start=target['start'] if target else None,end=target['end'] if target else None,score=target['score'] if target else None,position_control_start=ctrl['start'] if ctrl else None,position_control_status='available' if ctrl else 'unavailable',position_control_reason=None if ctrl else 'target unavailable or no nonoverlapping grid position'))
        controls.extend(control_rows)
    for group,key,maximize in [(trans,'score',True),(sustain,'score',True)]+[([r for r in controls if r['event']==name],'absolute_energy_difference',False) for name in ['transient','sustained']]:
        remaining=[r for r in group if r['eligible']]
        for r in group:r['selection_rank']=None
        rank=1
        while remaining:
            chosen=choose(remaining,key,maximize);chosen['selection_rank']=rank;rank+=1;remaining.remove(chosen)
    for e in events:e['control_shared_with']=[other['event'] for other in events if other is not e and e['position_control_start'] is not None and e['position_control_start']==other['position_control_start']]
    return events,dict(transient=trans,sustained=sustain,position_controls=controls)

def mask(family):
    import numpy as np
    z=np.arange(160,dtype=np.float64)/159
    if family=='cosine':z=.5*(1-np.cos(np.pi*z))
    elif family!='linear':raise ValueError('Unknown mask family')
    return np.concatenate([z,np.ones(960),z[::-1]])

def quantize(y):
    import numpy as np
    q=np.rint(y*32768);prefloat=int(np.count_nonzero((y< -1)|(y>32767/32768)|~np.isfinite(y)));rounded=int(np.count_nonzero((q< -32768)|(q>32767)|~np.isfinite(q)))
    result=dict(status='unsafe' if prefloat or rounded else 'safe',float_out_of_range=prefloat,rounded_out_of_range=rounded,pcm_payload_sha256=None,max_quantization_error=None,minimum=float(y.min()) if np.isfinite(y).all() else None,maximum=float(y.max()) if np.isfinite(y).all() else None)
    if result['status']=='safe':
        payload=q.astype('<i2').tobytes();result.update(pcm_payload_sha256=hashlib.sha256(payload).hexdigest(),max_quantization_error=float(np.max(np.abs(q/32768-y))))
    return result

def local_description(a):
    import numpy as np
    power=np.abs(np.fft.rfft(a,n=len(a)))**2;total=float(power.sum());freq=np.fft.rfftfreq(len(a),1/16000);high=(freq>=2000)&(freq<=7500)
    chunks=[a[i:i+160] for i in range(0,len(a),160)]
    return dict(energy=float(np.dot(a,a)),mean=float(a.mean()),rms=float(np.sqrt(np.mean(a*a))),abspeak=float(np.max(np.abs(a))),high_frequency_power_ratio=float(power[high].sum()/total) if total>0 else None,high_frequency_reason=None if total>0 else 'zero spectral power',envelope_rms_10ms=[float(np.sqrt(np.mean(c*c))) for c in chunks])

def operation(s,x,event,family,site):
    import numpy as np
    start=event['start'] if site=='target' else event['position_control_start'];row=dict(event=event['event'],family=family,site=site,start=start,end=start+1280 if start is not None else None,status='unavailable' if start is None else 'computed',reason='event or site unavailable' if start is None else None)
    if start is None:return row
    m=mask(family);delta=np.zeros_like(s);delta[start:start+1280]=-.75*m*s[start:start+1280];s0=s+delta;x0=x+delta;n=x-s;outside=np.ones(len(s),dtype=bool);outside[start:start+1280]=False
    qs=quantize(s0);qx=quantize(x0);qn=quantize(n);safe=qs['status']==qx['status']=='safe';quant_recon=None
    if safe:quant_recon=float(np.max(np.abs(np.rint(s0*32768)/32768+n-np.rint(x0*32768)/32768)))
    independently_quantized_residual_recon=None
    if safe and qn['status']=='safe':independently_quantized_residual_recon=float(np.max(np.abs(np.rint(s0*32768)/32768+np.rint(n*32768)/32768-np.rint(x0*32768)/32768)))
    segment=s[start:start+1280];delta_segment=delta[start:start+1280];row.update(status='safe' if safe else 'unsafe',reason=None if safe else 'prequantization or rounded PCM out of range; no clipping',mask_sha256=hashlib.sha256(m.astype('<f8').tobytes()).hexdigest(),delta_energy=float(np.dot(delta,delta)),untapered_clean_energy=float(np.dot(segment,segment)),clean_local=local_description(segment),delta_local=local_description(delta_segment),float_residual_reconstruction_max_error=float(np.max(np.abs(s0+n-x0))),applied_delta_max_error=float(np.max(np.abs((x0-x)-delta))),maskoutside_clean_equal=bool(np.array_equal(s0[outside],s[outside])),maskoutside_noisy_equal=bool(np.array_equal(x0[outside],x[outside])),clean_edited=qs,noisy_edited=qx,raw_residual_quantization=qn,quantized_arms_raw_residual_reconstruction_max_error=quant_recon,independently_quantized_residual_reconstruction_max_error=independently_quantized_residual_recon,clean_delta_energy_difference_vs_target=None)
    return row

def pair(s,x,raw_clean_bytes,raw_noisy_bytes):
    rows=features(s);events,candidates=select(s,rows);operations=[]
    for event in events:
        for family in ['cosine','linear']:
            group=[operation(s,x,event,family,site) for site in ['target','position_control']]
            if group[0]['status']!='unavailable' and group[1]['status']!='unavailable':
                difference=group[1]['delta_energy']-group[0]['delta_energy']
                for r in group:r['clean_delta_energy_difference_vs_target']=0. if r['site']=='target' else difference
            operations.extend(group)
    originals={}
    for role,data in [('clean',raw_clean_bytes),('noisy',raw_noisy_bytes)]:
        original=hashlib.sha256(data).hexdigest();identity=hashlib.sha256(bytes(memoryview(data))).hexdigest();originals[role]=dict(original_full_wav_sha256=original,identity_sham_full_wav_sha256=identity,byte_identical=(data==bytes(memoryview(data))))
    return dict(frames=rows,candidates=candidates,events=events,operations=operations,original_sham=originals)
