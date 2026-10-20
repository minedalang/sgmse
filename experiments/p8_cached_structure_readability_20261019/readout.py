"""Fixed cached-frame continuous-target readability.16linear fits, not zero training.
No StoRM training, no phoneme/H1H0/CAEG claim. Import does not read true sources."""
import os
for k in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS']:os.environ[k]='1'
import numpy as np

def representation(audio):
    length=320;hop=160;window=.5-.5*np.cos(2*np.pi*np.arange(length)/319)
    frames=np.stack([audio[start:start+length] for start in range(0,len(audio)-length+1,hop)])
    transform=np.fft.rfft(frames*window,n=512,axis=1);weights=np.ones(257)*2;weights[[0,-1]]=1
    power=np.abs(transform)**2*weights;energy=power.sum(axis=1);magnitude=np.sqrt(weights)*np.abs(transform)
    fractions=np.divide(power[:,128:241].sum(axis=1),energy,out=np.zeros(len(frames)),where=energy!=0)
    previous=np.concatenate([magnitude[:1],magnitude[:-1]],axis=0);den=np.linalg.norm(magnitude,axis=1)+np.linalg.norm(previous,axis=1)
    flux=np.divide(np.linalg.norm(np.maximum(magnitude[:,128:241]-previous[:,128:241],0),axis=1),den,out=np.zeros(len(frames)),where=den!=0)
    target=np.column_stack([fractions,flux]);band=np.column_stack([power[:,indices].sum(axis=1) for indices in np.array_split(np.arange(257),16)])
    band=np.divide(band,energy[:,None],out=np.zeros_like(band),where=energy[:,None]!=0)
    frame_power=energy/(512*float(np.dot(window,window)));log=np.log10(np.maximum(frame_power,1e-12));base=np.column_stack([band,log])
    context=np.concatenate([base[np.clip(np.arange(len(base))+offset,0,len(base)-1)] for offset in [-2,-1,0,1,2]],axis=1)
    full=np.column_stack([context,np.tile(base.mean(axis=0),(len(base),1)),np.tile(base.std(axis=0,ddof=0),(len(base),1))])
    simple0=np.column_stack([log,fractions,flux]);simple=np.column_stack([simple0,np.tile(simple0.mean(axis=0),(len(base),1))])
    assert full.shape==(len(frames),119) and simple.shape==(len(frames),6)
    return dict(full=full,simple=simple,identity=target,unused_tail_samples=len(audio)-(160*(len(frames)-1)+320),frames=len(frames))

def fit(train_x,train_y,lam=.01):
    weights=np.concatenate([np.full(len(x),1/(len(train_x)*len(x))) for x in train_x]);X=np.concatenate(train_x);Y=np.concatenate(train_y)
    mean=np.sum(weights[:,None]*X,axis=0);constant=np.all(X==X[0],axis=0);mean[constant]=X[0,constant];variance=np.sum(weights[:,None]*(X-mean)**2,axis=0);std=np.sqrt(variance);std=np.where(std==0,1,std)
    Z=np.column_stack([np.ones(len(X)),(X-mean)/std]);penalty=np.eye(Z.shape[1])*lam;penalty[0,0]=0
    coefficients=np.linalg.solve(Z.T@(weights[:,None]*Z)+penalty,Z.T@(weights[:,None]*Y))
    return dict(mean=mean,std=std,coefficients=coefficients,weight_sum=float(weights.sum()),train_target_mean=np.sum(weights[:,None]*Y,axis=0),lambda_=lam)

def predict(x,model):return np.column_stack([np.ones(len(x)),(x-model['mean'])/model['std']])@model['coefficients']

def evaluate(cases,check=lambda:None):
    speakers=sorted({r['speaker'] for r in cases});assert len(speakers)==4 and len(cases)==32
    fits=[];predictions={};rows=[]
    for speaker in speakers:
        train=[r for r in cases if r['speaker']!=speaker];held=[r for r in cases if r['speaker']==speaker];assert len(train)==24 and len(held)==8
        const=sum((r['target'].mean(axis=0) for r in train))/len(train)
        for arm in ['full_noisy','full_frontend','simple_noisy','simple_frontend']:
            check();kind,source=arm.split('_',1);model=fit([r[source][kind] for r in train],[r['target'] for r in train]);fits.append(dict(speaker=speaker,arm=arm,train_cases=[r['basename'] for r in train],heldout_cases=[r['basename'] for r in held],mean=model['mean'].tolist(),std=model['std'].tolist(),coefficients=model['coefficients'].tolist(),train_target_mean=model['train_target_mean'].tolist(),train_case_frames={r['basename']:len(r['target']) for r in train},weight_sum=model['weight_sum'],ridge_lambda=.01))
            for r in held:predictions[r['basename']+'__'+arm]=predict(r[source][kind],model)
        for r in held:
            for source in ['noisy','frontend']:predictions[r['basename']+'__identity_'+source]=r[source]['identity'].copy()
            predictions[r['basename']+'__train_mean_constant']=np.tile(const,(len(r['target']),1))
            predictions[r['basename']+'__target']=r['target']
    for r in cases:
        record=dict(basename=r['basename'],id=r['id'],speaker=r['speaker'],noise=r['noise'],snr_db=r['snr_db'],frames=len(r['target']),unused_tail_samples=r['tail'],scores={})
        for arm in ['full_noisy','full_frontend','simple_noisy','simple_frontend','identity_noisy','identity_frontend','train_mean_constant']:
            p=predictions[r['basename']+'__'+arm];err=(p-r['target'])**2;record['scores'][arm]=dict(MSE=err.mean(axis=0).tolist(),prediction_below0_fraction=(p<0).mean(axis=0).tolist(),prediction_above1_fraction=(p>1).mean(axis=0).tolist())
        rows.append(record)
    summaries={}
    for label,subset in [('all',rows)]+[('speaker_'+s,[r for r in rows if r['speaker']==s]) for s in speakers]+[('snr_'+str(q),[r for r in rows if r['snr_db']==q]) for q in [-18,0]]:
        arms={a:np.mean([r['scores'][a]['MSE'] for r in subset],axis=0).tolist() for a in rows[0]['scores']};differences={}
        for source in ['noisy','frontend']:
            for baseline in ['simple_'+source,'identity_'+source,'train_mean_constant']:differences['full_'+source+'_minus_'+baseline]=(np.asarray(arms['full_'+source])-arms[baseline]).tolist()
        differences['full_noisy_minus_full_frontend']=(np.asarray(arms['full_noisy'])-arms['full_frontend']).tolist();differences['simple_noisy_minus_simple_frontend']=(np.asarray(arms['simple_noisy'])-arms['simple_frontend']).tolist()
        summaries[label]=dict(case_n=len(subset),case_equal_MSE=arms,paired_MSE_differences=differences)
    return dict(fits=fits,cases=rows,summary=summaries,linear_fit_count=len(fits),context_column_order=[-2,-1,0,1,2]),predictions
