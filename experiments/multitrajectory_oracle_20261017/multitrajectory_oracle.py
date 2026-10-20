#!/usr/bin/env python3
from __future__ import annotations
import argparse,csv,hashlib,json,sys
from pathlib import Path
import numpy as np, soundfile as sf, torch, torch.nn.functional as F
from pesq import pesq
from pystoi import stoi
from torchmetrics.functional.audio.dnsmos import deep_noise_suppression_mean_opinion_score
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from sgmse.util.other import pad_spec

def load_wav(path,device):
 w,sr=sf.read(path,dtype='float32',always_2d=False); w=np.asarray(w); w=w.mean(-1) if w.ndim>1 else w
 if sr!=16000: raise ValueError(path)
 return torch.from_numpy(w).to(device),sr

def match_spec(x,target):
 out=x
 for dim in (-2,-1):
  n=target.shape[dim]; cur=out.shape[dim]
  if cur>n:
   st=(cur-n)//2; sl=[slice(None)]*out.ndim; sl[dim]=slice(st,st+n); out=out[tuple(sl)]
  elif cur<n:
   bef=(n-cur)//2; aft=n-cur-bef; pads=[0,0,0,0]
   if dim==-1:pads=[bef,aft,0,0]
   else:pads=[0,0,bef,aft]
   out=F.pad(out,pads)
 return out

def prepare_specs(model,noisy,clean,device):
 y=noisy[None,None,:].to(device); c=clean[None,None,:].to(device); norm=y.abs().amax().clamp_min(1e-12)
 y=y/norm; ypre=model._frontend_forward(y)
 if ypre.shape[-1]>y.shape[-1]: ypre=ypre[...,:y.shape[-1]]
 elif ypre.shape[-1]<y.shape[-1]: ypre=F.pad(ypre,(0,y.shape[-1]-ypre.shape[-1]))
 if model.train_type=='mode2': main,cond=ypre[:,0],y[:,0]
 else: main,cond=y[:,0],ypre[:,0]
 Y=pad_spec(torch.unsqueeze(model._forward_transform(model._stft(main)),0)); C=pad_spec(torch.unsqueeze(model._forward_transform(model._stft(cond)),0))
 X=pad_spec(torch.unsqueeze(model._forward_transform(model._stft((c/norm)[:,0])),0)); X=match_spec(X,Y)
 return Y,C,norm,int(noisy.numel())

def sisdr(ref,est):
 ref=ref-ref.mean(); est=est-est.mean(); p=np.dot(est,ref)/(np.dot(ref,ref)+1e-10)*ref; q=est-p
 return float(10*np.log10((np.dot(p,p)+1e-10)/(np.dot(q,q)+1e-10)))

def metrics(ref,est):
 L=min(len(ref),len(est)); r=ref[:L].astype('float32'); e=est[:L].astype('float32')
 out={'pesq':float(pesq(16000,r,e,'wb')),'stoi':float(stoi(r,e,16000,extended=False)),'sisdr':sisdr(r,e)}
 with torch.no_grad(): m=deep_noise_suppression_mean_opinion_score(torch.from_numpy(e),fs=16000,personalized=False,device='cpu',cache_session=True).numpy().reshape(-1)
 for k,v in zip(('p808','sig','bak','ovrl'),m): out['dnsmos_'+k]=float(v)
 return out

def run(args):
 out=Path(args.out_dir); out.mkdir(parents=True,exist_ok=True); wd=Path(args.wav_dir); wd.mkdir(parents=True,exist_ok=True)
 from sgmse.joint_data_module import JointWaveDataModule
 from sgmse.joint_model import JointSemambaSgmseModel
 model=JointSemambaSgmseModel.load_from_checkpoint(args.ckpt,map_location='cpu',strict=True,data_module_cls=JointWaveDataModule,base_dir='.',batch_size=1,num_workers=0,dummy=True,weights_only=False)
 model.eval(no_ema=not args.use_ema).to(args.device)
 all_rows=list(csv.DictReader(open(args.manifest))); rows=all_rows[args.start_index:args.start_index+args.max_files]; seeds=[2026,2027,2028,2029]; records=[]
 for i,row in enumerate(rows,1):
  noisy,_=load_wav(row['noisy'],args.device); clean,_=load_wav(row['clean'],args.device); L=min(noisy.numel(),clean.numel()); noisy=noisy[:L]; clean=clean[:L]
  Y,C,norm,length=prepare_specs(model,noisy,clean,args.device); cal=model.build_rrsd_score_calibrator(Y,C,scale=1.0); score_fn=lambda x,y,t:model(x,y,t,y_cond=C)
  for seed in seeds:
   torch.manual_seed(seed); sampler=model.get_pc_sampler('reverse_diffusion','none',Y,score_fn=score_fn,N=args.N,snr=args.snr,score_calibrator=cal); sample,nfe=sampler(); wav=(model.to_audio(sample.squeeze(),length=length).detach().cpu()*float(norm.cpu())).numpy(); p=wd/f"{Path(row['filename']).stem}_seed{seed}.wav"; sf.write(p,np.clip(wav,-1,1),16000)
   d={'filename':row['filename'],'seed':seed,'wav':str(p),'nfe':int(nfe)}; d.update(metrics(clean[:len(wav)].detach().cpu().numpy(),wav)); records.append(d); print(i,seed,flush=True)
 # Oracle: max OVRL with SI-SDR within 0.05 dB of seed2026 baseline per file
 by={}
 for d in records: by.setdefault(d['filename'],[]).append(d)
 oracle=[]; baseline=[]
 for fn,ds in by.items():
  base=next(d for d in ds if d['seed']==2026); baseline.append(base); eligible=[d for d in ds if d['sisdr']>=base['sisdr']-0.05]; sel=max(eligible,key=lambda d:d['dnsmos_ovrl']); z=dict(sel); z['oracle_source_seed']=sel['seed']; z['baseline_seed2026_sisdr']=base['sisdr']; z['eligible_count']=len(eligible); oracle.append(z)
 rng=np.random.default_rng(2026); metrics_names=['pesq','stoi','sisdr','dnsmos_p808','dnsmos_sig','dnsmos_bak','dnsmos_ovrl']; paired={}
 for m in metrics_names:
  diff=np.array([z[m]-b[m] for z,b in zip(oracle,baseline)]); draw=rng.choice(diff,size=(20000,len(diff)),replace=True).mean(1); paired[m]={'delta':float(diff.mean()),'ci95':[float(x) for x in np.percentile(draw,[2.5,97.5])]}
 means={k:float(np.mean([d[k] for d in baseline])) for k in metrics_names}; omeans={k:float(np.mean([d[k] for d in oracle])) for k in metrics_names}
 result={'metadata':{'n_files':len(rows),'seeds':seeds,'N':args.N,'snr':args.snr,'wav_dir':str(wd),'bootstrap':20000,'bootstrap_seed':2026,'selection':'max DNSMOS OVRL subject to SI-SDR >= seed2026 baseline - 0.05 dB'},'baseline_seed2026_means':means,'oracle_means':omeans,'paired_bootstrap_oracle_vs_seed2026':paired,'per_trajectory':records,'oracle_per_file':oracle}
 (out/'metrics.json').write_text(json.dumps(result,indent=2)+'\n'); print(json.dumps({'baseline':means,'oracle':omeans,'paired':paired},indent=2))
if __name__=='__main__':
 p=argparse.ArgumentParser(); p.add_argument('--ckpt',default='joint_train_logs/sb61i0gj_RRSD/epoch=118-step=80000.ckpt'); p.add_argument('--manifest',default='experiments/rrsd_cv_repeat32/manifest.csv'); p.add_argument('--out_dir',default='experiments/multitrajectory_oracle_20261017'); p.add_argument('--wav_dir',default='/dev/shm/multitraj_router_wavs_16'); p.add_argument('--max_files',type=int,default=16); p.add_argument('--start_index',type=int,default=0); p.add_argument('--N',type=int,default=30); p.add_argument('--snr',type=float,default=.5); p.add_argument('--device',default='cuda'); p.add_argument('--use_ema',action='store_true'); run(p.parse_args())
