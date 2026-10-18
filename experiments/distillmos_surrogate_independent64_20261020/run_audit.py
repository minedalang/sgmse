from __future__ import annotations
import csv,json,time,hashlib
from pathlib import Path
import numpy as np, torch, soundfile as sf
from scipy.stats import pearsonr,spearmanr
import distillmos
ROOT=Path('/data/coding/sgmse'); EXP=ROOT/'experiments/distillmos_surrogate_independent64_20261020'; BASE=ROOT/'experiment_logs/speaker_headroom_replication_64_20261017'; MET=ROOT/'experiments/speaker_headroom_replication_64_20261017/quality_per_utterance.csv'; DEV=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
def sha(p):
 h=hashlib.sha256();
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
def main():
 with open(MET,newline='') as f: rows=list(csv.DictReader(f))
 model=distillmos.ConvTransformerSQAModel(); model.eval().to(DEV); start=time.perf_counter(); out=[]
 for i,r in enumerate(rows):
  z={'filename':r['filename'],'speaker':r['speaker']}
  for arm in ('off','router'):
   p=BASE/arm/r['filename']
   wav,sr=sf.read(p,dtype='float32',always_2d=False)
   if sr!=16000: raise RuntimeError(sr)
   with torch.inference_mode(): q=model(torch.from_numpy(np.asarray(wav,dtype=np.float32))[None].to(DEV)).reshape(-1)[0].item()
   z[arm+'_distillmos']=float(q)
  z['distill_delta']=z['router_distillmos']-z['off_distillmos']
  z['si_sdr_delta']=float(r['router_si_sdr_db'])-float(r['off_si_sdr_db'])
  z['pesq_delta']=float(r['router_pesq_wb'])-float(r['off_pesq_wb'])
  z['stoi_delta']=float(r['router_stoi'])-float(r['off_stoi'])
  z['ovrl_delta']=float(r['router_dnsmos_ovrl'])-float(r['off_dnsmos_ovrl'])
  z['sig_delta']=float(r['router_dnsmos_sig'])-float(r['off_dnsmos_sig'])
  out.append(z)
 fields=list(out[0]);
 with (EXP/'per_file.csv').open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(out)
 def corr(a,b): return {'pearson':float(pearsonr(a,b).statistic),'spearman':float(spearmanr(a,b).statistic)}
 d={k:corr([x['distill_delta'] for x in out],[x[k] for x in out]) for k in ('si_sdr_delta','pesq_delta','stoi_delta','ovrl_delta','sig_delta')}
 summary={'experiment_id':'distillmos_surrogate_independent64_20261020','status':'completed_read_only','device':str(DEV),'n_utterances':len(out),'n_speakers':len(set(x['speaker'] for x in out)),'runtime_seconds':time.perf_counter()-start,'delta_correlations':d,'means':{k:float(np.mean([x[k] for x in out])) for k in ('distill_delta','si_sdr_delta','pesq_delta','stoi_delta','ovrl_delta','sig_delta')},'source_metrics':str(MET),'training':False,'new_inference':False,'audio_copied':False,'limitations':['independent CV relative to prior 32-utterance development split but not an independent human listening set','DNSMOS and Distill-MOS are both learned metrics','correlation is descriptive and does not establish causal or human preference alignment']}
 (EXP/'metrics.json').write_text(json.dumps(summary,indent=2)+'\n'); print(json.dumps(summary,indent=2))
if __name__=='__main__':main()
