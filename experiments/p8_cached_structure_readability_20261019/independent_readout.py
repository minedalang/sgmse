"""Independent feature bookkeeping + weighted augmented least squares.
Shares NumPy FFT backend and cache IO lifecycle, imports no main business functions."""
import os
for k in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS']:os.environ[k]='1'
import numpy as np,math

def representation(y):
    hann=.5-.5*np.cos(2*np.pi*np.arange(320)/319);factor=np.r_[1.,np.full(255,2.),1.];groups=np.array_split(np.arange(257),16)
    bases=[];simple=[];targets=[];last=None
    for start in range(0,len(y)-319,160):
        spectrum=np.fft.rfft(y[start:start+320]*hann,512);energy=factor*np.abs(spectrum)**2;total=float(energy.sum());m=np.sqrt(factor)*np.abs(spectrum)
        fraction=float(energy[128:241].sum()/total) if total!=0 else 0.
        previous=m if last is None else last;den=float(np.linalg.norm(m)+np.linalg.norm(previous));flux=float(np.linalg.norm(np.maximum(m[128:241]-previous[128:241],0))/den) if den!=0 else 0.;last=m
        log=float(np.log10(max(total/(512*float(hann@hann)),1e-12)));bases.append([float(energy[g].sum()/total) if total!=0 else 0. for g in groups]+[log]);simple.append([log,fraction,flux]);targets.append([fraction,flux])
    base=np.asarray(bases);local=np.asarray(simple);full=[];simple_out=[]
    for j in range(len(base)):
        full.append(np.concatenate([base[min(max(j+d,0),len(base)-1)] for d in [-2,-1,0,1,2]]+[base.mean(axis=0),base.std(axis=0,ddof=0)]))
        simple_out.append(np.r_[local[j],local.mean(axis=0)])
    return dict(full=np.asarray(full),simple=np.asarray(simple_out),identity=np.asarray(targets),unused_tail_samples=len(y)-(160*(len(base)-1)+320),frames=len(base))

def fit(xx,yy,lam=.01):
    X=np.vstack(xx);Y=np.vstack(yy);w=np.asarray([1/(len(xx)*len(case)) for case in xx for _ in range(len(case))]);mu=np.sum(X*w[:,None],axis=0);constant=np.all(X==X[0],axis=0);mu[constant]=X[0,constant]
    sigma=np.sqrt(np.sum(w[:,None]*(X-mu)**2,axis=0));sigma[sigma==0]=1
    Z=np.c_[np.ones(len(X)),(X-mu)/sigma];regularizer=np.zeros((X.shape[1],X.shape[1]+1));regularizer[:,1:]=np.eye(X.shape[1])*math.sqrt(lam)
    augmented=np.vstack([Z*np.sqrt(w)[:,None],regularizer]);response=np.vstack([Y*np.sqrt(w)[:,None],np.zeros((X.shape[1],Y.shape[1]))]);coeff=np.linalg.lstsq(augmented,response,rcond=None)[0]
    return dict(mean=mu,std=sigma,coefficients=coeff,weight_sum=float(w.sum()),train_target_mean=np.sum(w[:,None]*Y,axis=0),lambda_=lam)

def evaluate(cases,check=lambda:None):
    speakers=sorted(set(r['speaker'] for r in cases));assert len(cases)==32 and len(speakers)==4
    fits=[];output={};arms=['full_noisy','full_frontend','simple_noisy','simple_frontend'];allarms=arms+['identity_noisy','identity_frontend','train_mean_constant']
    for heldspeaker in speakers:
        train=[r for r in cases if r['speaker']!=heldspeaker];held=[r for r in cases if r['speaker']==heldspeaker];assert len(train)==24 and len(held)==8
        constant=np.mean(np.asarray([r['target'].mean(axis=0) for r in train]),axis=0)
        for arm in arms:
            check();feature,source=arm.split('_');m=fit([r[source][feature] for r in train],[r['target'] for r in train]);fits.append(dict(speaker=heldspeaker,arm=arm,train_cases=[r['basename'] for r in train],heldout_cases=[r['basename'] for r in held],mean=m['mean'].tolist(),std=m['std'].tolist(),coefficients=m['coefficients'].tolist(),train_target_mean=m['train_target_mean'].tolist(),train_case_frames={r['basename']:len(r['target']) for r in train},weight_sum=m['weight_sum'],ridge_lambda=.01))
            for r in held:output[r['basename']+'__'+arm]=np.c_[np.ones(len(r['target'])),(r[source][feature]-m['mean'])/m['std']]@m['coefficients']
        for r in held:
            output[r['basename']+'__target']=r['target'];output[r['basename']+'__train_mean_constant']=np.repeat(constant[None,:],len(r['target']),axis=0)
            for source in ['noisy','frontend']:output[r['basename']+'__identity_'+source]=r[source]['identity']
    rows=[]
    for r in cases:
        scores={}
        for arm in allarms:
            values=output[r['basename']+'__'+arm];scores[arm]=dict(MSE=np.mean(np.square(values-r['target']),axis=0).tolist(),prediction_below0_fraction=np.mean(values<0,axis=0).tolist(),prediction_above1_fraction=np.mean(values>1,axis=0).tolist())
        rows.append(dict(basename=r['basename'],id=r['id'],speaker=r['speaker'],noise=r['noise'],snr_db=r['snr_db'],frames=len(r['target']),unused_tail_samples=r['tail'],scores=scores))
    subsets={'all':rows}
    subsets.update({'speaker_'+s:[r for r in rows if r['speaker']==s] for s in speakers});subsets.update({'snr_'+str(q):[r for r in rows if r['snr_db']==q] for q in [-18,0]});summaries={}
    for name,subset in subsets.items():
        means={a:np.mean([r['scores'][a]['MSE'] for r in subset],axis=0).tolist() for a in allarms};diff={}
        for source in ['noisy','frontend']:
            for baseline in ['simple_'+source,'identity_'+source,'train_mean_constant']:diff['full_'+source+'_minus_'+baseline]=(np.asarray(means['full_'+source])-means[baseline]).tolist()
        diff['full_noisy_minus_full_frontend']=(np.asarray(means['full_noisy'])-means['full_frontend']).tolist();diff['simple_noisy_minus_simple_frontend']=(np.asarray(means['simple_noisy'])-means['simple_frontend']).tolist()
        summaries[name]=dict(case_n=len(subset),case_equal_MSE=means,paired_MSE_differences=diff)
    return dict(fits=fits,cases=rows,summary=summaries,linear_fit_count=len(fits),context_column_order=[-2,-1,0,1,2]),output

def compare(a,b):
    if isinstance(a,dict):
        if set(a)!=set(b):raise AssertionError('Result key mismatch')
        for key in a:compare(a[key],b[key])
    elif isinstance(a,list):
        if len(a)!=len(b):raise AssertionError('Result length mismatch')
        for x,y in zip(a,b):compare(x,y)
    elif isinstance(a,(int,float)) and not isinstance(a,bool):
        if b is None or not math.isclose(a,b,rel_tol=1e-8,abs_tol=1e-9):raise AssertionError((a,b))
    elif a!=b:raise AssertionError((a,b))
