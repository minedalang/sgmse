#!/usr/bin/env python3
from __future__ import annotations
import csv, hashlib, json, time
from pathlib import Path
import numpy as np
import torch, soundfile as sf
from scipy.stats import spearmanr, pearsonr
import distillmos

ROOT=Path('/data/coding/sgmse')
EXP=ROOT/'experiments/distillmos_surrogate_audit_20261020'
MET=ROOT/'experiments/perceptual_representation_probe_20261017/per_file.csv'
BASE=ROOT/'experiment_logs/rrsd_tcut_screen'
SEL=ROOT/'experiment_logs/rrsd_selective_screen/selective'
ARMS={'noisy':BASE/'noisy','off':BASE/'off','router':BASE/'router','cut005':BASE/'cut005','selective':SEL}
DEVICE=torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def sha(path):
 h=hashlib.sha256();
 with open(path,'rb') as f:
  for b in iter(lambda:f.read(1<<20),b''): h.update(b)
 return h.hexdigest()

def load_rows():
 with MET.open(newline='') as f: return list(csv.DictReader(f))

def run_model(paths):
 model=distillmos.ConvTransformerSQAModel(); model.eval().to(DEVICE)
 out={}; start=time.perf_counter()
 for i,p in enumerate(paths):
  wav,sr=sf.read(p,dtype='float32',always_2d=False)
  if sr!=16000: raise RuntimeError(f'bad sr {sr}: {p}')
  x=torch.from_numpy(np.asarray(wav,dtype=np.float32)).reshape(1,-1).to(DEVICE)
  with torch.inference_mode(): y=model(x)
  out[str(p)]=float(y.reshape(-1)[0].detach().cpu())
  if (i+1)%20==0: print(f'processed {i+1}/{len(paths)}',flush=True)
 return out,time.perf_counter()-start

def rank_agreement(recs,arms,metric):
 vals=[]
 for fn in sorted(recs):
  rr=recs[fn]; pairs=[]
  for i,a in enumerate(arms):
   for b in arms[i+1:]:
    q1,q2=rr[a].get(metric),rr[b].get(metric); m1,m2=rr[a]['distillmos'],rr[b]['distillmos']
    if q1 is None or q2 is None or not np.isfinite([q1,q2,m1,m2]).all() or q1==q2: continue
    pairs.append(float((m1-m2)*(q1-q2)>0))
  vals.append(float(np.mean(pairs)) if pairs else np.nan)
 return np.array(vals,float)

def bootstrap_mean(x,n=10000,seed=20261020):
 x=np.asarray(x,float); x=x[np.isfinite(x)];
 if not len(x): return [None,None,None]
 rng=np.random.default_rng(seed); boot=x[rng.integers(0,len(x),size=(n,len(x)))].mean(1)
 return [float(x.mean()),float(np.quantile(boot,.025)),float(np.quantile(boot,.975))]

def main():
 rows=load_rows(); fns=sorted({r['filename'] for r in rows})
 allpaths=[ARMS[a]/fn for fn in fns for a in ARMS]
 for p in allpaths:
  if not p.is_file(): raise FileNotFoundError(p)
 mos,rt=run_model(allpaths)
 recs={fn:{'filename':fn} for fn in fns}
 for fn in fns:
  for a in ARMS: recs[fn][a]={'distillmos':mos[str(ARMS[a]/fn)]}
 for r in rows:
  fn=r['filename']; a=r['arm']; d=recs[fn][a]
  for k in ('pesq','stoi','si_sdr','dnsmos_sig','dnsmos_ovrl','dnsmos_bak'):
   v=r.get(k,'')
   d[k]=float(v) if v not in ('',None) else None
 outrows=[]
 for fn in fns:
  for a in ARMS:
   z=recs[fn][a].copy(); z['arm']=a; outrows.append(z)
 with (EXP/'per_file.csv').open('w',newline='') as f:
  fields=['filename','arm','distillmos','pesq','stoi','si_sdr','dnsmos_sig','dnsmos_ovrl','dnsmos_bak']; w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(outrows)
 rank={}
 quality_arms=['noisy','off','router','cut005','selective']
 dns_arms=['router','cut005','selective']
 for metric in ('pesq','stoi','si_sdr'):
  x=rank_agreement(recs,quality_arms,metric); rank[f'all_{metric}']=bootstrap_mean(x)
 for metric in ('dnsmos_sig','dnsmos_ovrl','dnsmos_bak'):
  x=rank_agreement(recs,dns_arms,metric); rank[f'dns_{metric}']=bootstrap_mean(x)
 corr={}
 for metric,arms in [('pesq',quality_arms),('stoi',quality_arms),('si_sdr',quality_arms),('dnsmos_sig',dns_arms),('dnsmos_ovrl',dns_arms),('dnsmos_bak',dns_arms)]:
  xx=[]; yy=[]
  for fn in fns:
   for i,a in enumerate(arms):
    for b in arms[i+1:]:
     q1,q2=recs[fn][a].get(metric),recs[fn][b].get(metric)
     if q1 is None or q2 is None: continue
     xx.append(recs[fn][a]['distillmos']-recs[fn][b]['distillmos']); yy.append(q1-q2)
  corr[metric]={'n_pairs':len(xx),'pearson_r':float(pearsonr(xx,yy).statistic) if len(xx)>2 else None,'spearman_r':float(spearmanr(xx,yy).statistic) if len(xx)>2 else None}
 summary={'experiment_id':'distillmos_surrogate_audit_20261020','status':'completed_read_only','device':str(DEVICE),'n_utterances':len(fns),'arms':list(ARMS),'model':'distillmos.ConvTransformerSQAModel variant 7','weights_sha256':sha('/data/coding/sgmse/env/lib/python3.10/site-packages/distillmos/weights/distill_mos_v7.pt'),'runtime_seconds':rt,'ranking_pair_agreement_mean_ci95':rank,'pair_delta_correlations':corr,'source_metric_file':str(MET),'audio_copied':False,'training':False,'dataset_modified':False,'checkpoint_modified':False,'limitations':['existing 32-utterance development split reused; not independent validation','Distill-MOS is a learned quality surrogate, not human MOS ground truth','DNSMOS and Distill-MOS are both model-based and are not independent human validation','ranking only tests whether this surrogate is safe enough to screen a future adapter; no adapter was trained']}
 (EXP/'metrics.json').write_text(json.dumps(summary,indent=2)+'\n')
 print(json.dumps(summary,indent=2))
if __name__=='__main__': main()
