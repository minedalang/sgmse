"""Independent business reconstruction. Uses same NumPy FFT/cos backend for exact
float/PCM/mask identity but separate framing/selection/control/operation code.
No import of event_integrity or its selection/mask/quantization functions."""
import hashlib,math,os
for name in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS']:os.environ[name]='1'

def independent_pair(s,x,clean_bytes,noisy_bytes):
    import numpy as np
    size=len(s);window=np.asarray(.5-.5*np.cos((2*np.pi*np.arange(320))/319));spectra=[];frames=[]
    for begin in range(0,size-320+1,160):
        transformed=np.fft.rfft(s[begin:begin+320]*window,512);p=np.abs(transformed)**2;den=float(np.sum(p));spectra.append(p/den if den>0 and math.isfinite(den) else None)
    for index,q in enumerate(spectra):
        ratio=float(np.sum(q[64:241])) if q is not None else None
        flux=float(np.sum(np.maximum(q[64:241]-spectra[index-1][64:241],0))) if index and q is not None and spectra[index-1] is not None else None
        frames.append(dict(frame=index,start=index*160,R=ratio,F=flux,Q_reason=None if q is not None else 'nonpositive or nonfinite spectral power',F_reason=None if flux is not None else 'first frame or undefined neighboring Q'))
    def pick(candidates,field,ascending=False):
        pool=[r for r in candidates if r['eligible']]
        if not pool:return None
        values=[r[field] for r in pool];extreme=min(values) if ascending else max(values)
        starts=sorted(r['start'] for r in pool if abs(r[field]-extreme)<=1e-12)
        return next(r for r in pool if r['start']==starts[0])
    def disjoint(a,b):return min(a,b)+1440<=max(a,b)
    transient=[]
    for f in frames:
        begin=f['start']-480
        if begin<3200 or begin+1280>size-3200:continue
        eligible=f['F'] is not None and f['F']>0
        transient.append(dict(start=begin,end=begin+1280,score=f['F'],eligible=eligible,reason=None if eligible else 'no positive finite F'))
    t=pick(transient,'score');sustained=[];grid=range(3200,size-3200-1280+1,160)
    for begin in grid:
        rlist=[frames[begin//160+k]['R'] for k in range(7)];score=sum(rlist)/7 if None not in rlist else None
        eligible=score is not None and score>0 and (t is None or disjoint(begin,t['start']))
        sustained.append(dict(start=begin,end=begin+1280,score=score,eligible=eligible,reason=None if eligible else 'undefined/nonpositive R or overlap transient'))
    u=pick(sustained,'score');selected=[r['start'] for r in [t,u] if r is not None];events=[];controls=[]
    for label,target in [('transient',t),('sustained',u)]:
        candidates=[]
        if target is not None:
            excerpt=s[target['start']:target['end']];energy=float(excerpt@excerpt)
            for begin in grid:
                fragment=s[begin:begin+1280];en=float(fragment@fragment);eligible=all(disjoint(begin,k) for k in selected)
                candidates.append(dict(event=label,start=begin,end=begin+1280,energy=en,target_energy=energy,absolute_energy_difference=abs(en-energy),eligible=eligible,reason=None if eligible else 'overlap selected target plus gap'))
        chosen=pick(candidates,'absolute_energy_difference',True)
        events.append(dict(event=label,status='available' if target else 'unavailable',reason=None if target else 'no positive defined nonoverlapping candidate',start=target['start'] if target else None,end=target['end'] if target else None,score=target['score'] if target else None,position_control_start=chosen['start'] if chosen else None,position_control_status='available' if chosen else 'unavailable',position_control_reason=None if chosen else 'target unavailable or no nonoverlapping grid position'))
        controls.extend(candidates)
    for group,field,ascending in [(transient,'score',False),(sustained,'score',False)]+[([r for r in controls if r['event']==name],'absolute_energy_difference',True) for name in ['transient','sustained']]:
        left=[r for r in group if r['eligible']]
        for candidate in group:candidate['selection_rank']=None
        for rank in range(1,len(left)+1):
            winner=pick(left,field,ascending);winner['selection_rank']=rank;left.remove(winner)
    for event in events:event['control_shared_with']=[other['event'] for other in events if other is not event and event['position_control_start'] is not None and other['position_control_start']==event['position_control_start']]
    def quant(y):
        rounded=np.rint(y*32768);pre=sum((y< -1)|(y>32767/32768)|~np.isfinite(y));post=sum((rounded< -32768)|(rounded>32767)|~np.isfinite(rounded));safe=not (pre or post)
        return dict(status='safe' if safe else 'unsafe',float_out_of_range=int(pre),rounded_out_of_range=int(post),pcm_payload_sha256=hashlib.sha256(rounded.astype('<i2').tobytes()).hexdigest() if safe else None,max_quantization_error=float(np.max(np.abs(rounded/32768-y))) if safe else None,minimum=float(np.min(y)) if np.isfinite(y).all() else None,maximum=float(np.max(y)) if np.isfinite(y).all() else None)
    def describe(a):
        fourier=np.fft.rfft(a,len(a));p=np.abs(fourier)**2;freq=np.fft.rfftfreq(len(a),1/16000);den=float(np.sum(p));energy=float(a@a)
        return dict(energy=energy,mean=float(np.mean(a)),rms=float(np.sqrt(np.mean(a*a))),abspeak=float(np.max(np.abs(a))),high_frequency_power_ratio=float(np.sum(p[(freq>=2000)&(freq<=7500)])/den) if den>0 else None,high_frequency_reason=None if den>0 else 'zero spectral power',envelope_rms_10ms=[float(np.sqrt(np.mean(a[i:i+160]**2))) for i in range(0,len(a),160)])
    operations=[]
    for e in events:
        for family in ['cosine','linear']:
            ramp=np.arange(160,dtype=np.float64)/159
            if family=='cosine':ramp=.5*(1-np.cos(np.pi*ramp))
            m=np.r_[ramp,np.ones(960),ramp[::-1]];local=[]
            for site in ['target','position_control']:
                begin=e['start'] if site=='target' else e['position_control_start'];r=dict(event=e['event'],family=family,site=site,start=begin,end=begin+1280 if begin is not None else None,status='unavailable' if begin is None else 'computed',reason='event or site unavailable' if begin is None else None)
                if begin is None:local.append(r);continue
                delta=np.zeros(size,dtype=np.float64);delta[begin:begin+1280]=-.75*m*s[begin:begin+1280];a=s+delta;b=x+delta;noise=x-s;qs=quant(a);qx=quant(b);qn=quant(noise);safe=qs['status']==qx['status']=='safe';outside=np.r_[np.arange(begin),np.arange(begin+1280,size)]
                raw_recon=float(np.max(np.abs(np.rint(a*32768)/32768+noise-np.rint(b*32768)/32768))) if safe else None
                all_recon=float(np.max(np.abs(np.rint(a*32768)/32768+np.rint(noise*32768)/32768-np.rint(b*32768)/32768))) if safe and qn['status']=='safe' else None
                r.update(status='safe' if safe else 'unsafe',reason=None if safe else 'prequantization or rounded PCM out of range; no clipping',mask_sha256=hashlib.sha256(m.astype('<f8').tobytes()).hexdigest(),delta_energy=float(delta@delta),untapered_clean_energy=float(s[begin:begin+1280]@s[begin:begin+1280]),clean_local=describe(s[begin:begin+1280]),delta_local=describe(delta[begin:begin+1280]),float_residual_reconstruction_max_error=float(np.max(np.abs(a+noise-b))),applied_delta_max_error=float(np.max(np.abs((b-x)-delta))),maskoutside_clean_equal=bool(np.array_equal(a[outside],s[outside])),maskoutside_noisy_equal=bool(np.array_equal(b[outside],x[outside])),clean_edited=qs,noisy_edited=qx,raw_residual_quantization=qn,quantized_arms_raw_residual_reconstruction_max_error=raw_recon,independently_quantized_residual_reconstruction_max_error=all_recon,clean_delta_energy_difference_vs_target=None)
                local.append(r)
            if all(r['status']!='unavailable' for r in local):
                local[0]['clean_delta_energy_difference_vs_target']=0.;local[1]['clean_delta_energy_difference_vs_target']=local[1]['delta_energy']-local[0]['delta_energy']
            operations.extend(local)
    original={role:dict(original_full_wav_sha256=hashlib.sha256(data).hexdigest(),identity_sham_full_wav_sha256=hashlib.sha256(memoryview(data).tobytes()).hexdigest(),byte_identical=data==memoryview(data).tobytes()) for role,data in [('clean',clean_bytes),('noisy',noisy_bytes)]}
    return dict(frames=frames,candidates=dict(transient=transient,sustained=sustained,position_controls=controls),events=events,operations=operations,original_sham=original)

def compare(a,b):
    if isinstance(a,dict):
        if set(a)!=set(b):raise AssertionError(('keys',set(a)^set(b)))
        for key,value in a.items():compare(value,b[key])
    elif isinstance(a,list):
        if len(a)!=len(b):raise AssertionError('List scope mismatch')
        for x,y in zip(a,b):compare(x,y)
    elif isinstance(a,(int,float)) and not isinstance(a,bool):
        if b is None or not math.isclose(a,b,rel_tol=1e-9,abs_tol=1e-12):raise AssertionError((a,b))
    elif a!=b:raise AssertionError((a,b))
