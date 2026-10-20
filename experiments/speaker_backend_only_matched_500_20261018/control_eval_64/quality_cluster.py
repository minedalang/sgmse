import csv,json
from pathlib import Path
import numpy as np,soundfile as sf
from pesq import pesq
from pystoi import stoi
R=Path('/data/coding/sgmse'); E=R/'experiments/speaker_backend_only_matched_500_20261018/control_eval_64'; M=R/'experiments/speaker_headroom_replication_64_20261017/manifest.csv'; paths={'control':E/'control','off':R/'experiment_logs/speaker_headroom_replication_64_20261017/off','router':R/'experiment_logs/speaker_headroom_replication_64_20261017/router'}
def sisdr(s,e):
 s=s-s.mean();e=e-e.mean();a=np.dot(e,s)/(np.dot(s,s)+1e-12);t=a*s;return 10*np.log10((np.dot(t,t)+1e-12)/(np.dot(e-t,e-t)+1e-12))
rows=list(csv.DictReader(M.open())); rec=[]
for r in rows:
 s,_=sf.read(r['clean_path'],dtype='float32'); z={'filename':r['filename'],'uttid':r['uttid'],'speaker':r['speaker']}
 for a,p in paths.items():
  e,_=sf.read(p/r['filename'],dtype='float32'); assert len(e)==len(s)==int(r['samples'])
  z[f'{a}_pesq_wb']=float(pesq(16000,s,e,'wb')); z[f'{a}_stoi']=float(stoi(s,e,16000,extended=False)); z[f'{a}_sisdr_db']=float(sisdr(s,e))
 rec.append(z)
with (E/'quality_per_utterance.csv').open('w',newline='') as f:
 w=csv.DictWriter(f,fieldnames=rec[0].keys());w.writeheader();w.writerows(rec)
sp=sorted({r['speaker'] for r in rec}); idx={s:[i for i,r in enumerate(rec) if r['speaker']==s] for s in sp}; rng=np.random.default_rng(20261017); draws=rng.integers(0,len(sp),size=(5000,len(sp))); counts=np.array([np.bincount(x,minlength=len(sp)) for x in draws])
out={'n':len(rec),'n_speakers':len(sp),'bootstrap':{'replicates':5000,'seed':20261017,'unit':'speaker cluster'},'metrics':{}}
for m in ('pesq_wb','stoi','sisdr_db'):
 q={}
 for a in paths:
  v=np.array([r[f'{a}_{m}'] for r in rec]); sv=np.array([v[idx[s]].mean() for s in sp]); b=(counts*sv).sum(1)/counts.sum(1);q[a]={'mean':float(v.mean()),'speaker_cluster_bootstrap_95ci':[float(np.quantile(b,.025)),float(np.quantile(b,.975))]}
 d=np.array([r[f'control_{m}']-r[f'off_{m}'] for r in rec]); sd=np.array([d[idx[s]].mean() for s in sp]); b=(counts*sd).sum(1)/counts.sum(1); q['control_minus_off']={'mean_delta':float(d.mean()),'speaker_cluster_bootstrap_95ci':[float(np.quantile(b,.025)),float(np.quantile(b,.975))]}
 d=np.array([r[f'control_{m}']-r[f'router_{m}'] for r in rec]); sd=np.array([d[idx[s]].mean() for s in sp]); b=(counts*sd).sum(1)/counts.sum(1); q['control_minus_router']={'mean_delta':float(d.mean()),'speaker_cluster_bootstrap_95ci':[float(np.quantile(b,.025)),float(np.quantile(b,.975))]}
 out['metrics'][m]=q
(E/'quality_cluster_metrics.json').write_text(json.dumps(out,indent=2)+'\n'); print(json.dumps(out,indent=2))
