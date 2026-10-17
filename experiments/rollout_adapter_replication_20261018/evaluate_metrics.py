import csv,json,time,sys
from pathlib import Path
import numpy as np,soundfile as sf
from pesq import pesq
from pystoi import stoi
sys.path.insert(0,'/data/coding/mymetrics'); import get_metrics_for_papertable_forDNS as dm
E=Path(__file__).parent; M=Path('/data/coding/sgmse/experiments/tr0_fresh_cv_manifest_20261018/manifest.csv'); out=[]
def sisdr(s,e):
 s=s-s.mean();e=e-e.mean(); a=np.dot(e,s)/(np.dot(s,s)+1e-12);t=a*s; return 10*np.log10((np.dot(t,t)+1e-12)/(np.dot(e-t,e-t)+1e-12))
rows=list(csv.DictReader(M.open()));
for i,r in enumerate(rows[:8]):
 clean,_=sf.read(r['clean'],dtype='float32'); z={'filename':r['filename'],'speaker':r['speaker'],'split':'train' if i<6 else 'holdout'}
 for a in ['baseline','adapter']:
  p=E/'audio'/a/r['filename']; e,sr=sf.read(p,dtype='float32'); z[f'{a}_pesq']=float(pesq(16000,clean,e,'wb'));z[f'{a}_stoi']=float(stoi(clean,e,16000,extended=False));z[f'{a}_sisdr']=float(sisdr(clean,e));
  try:z[f'{a}_dnsmos']=dm.calc_dnsmos(e,sr=sr,device='auto')
  except Exception as ex:z[f'{a}_dnsmos_error']=repr(ex)
 out.append(z)
(E/'per_utterance_metrics.json').write_text(json.dumps(out,indent=2,default=str)+'\n')
summary={'n':len(out),'metrics':{}}
for m in ['pesq','stoi','sisdr']:
 summary['metrics'][m]={a:float(np.mean([r[f'{a}_{m}'] for r in out])) for a in ['baseline','adapter']};summary['metrics'][m]['adapter_minus_baseline']=summary['metrics'][m]['adapter']-summary['metrics'][m]['baseline']
for a in ['baseline','adapter']:
 vals=[r.get(f'{a}_dnsmos') for r in out if isinstance(r.get(f'{a}_dnsmos'),dict)]; summary.setdefault('dnsmos',{})[a]={k:float(np.mean([v[k] for v in vals])) for k in vals[0]} if vals else {'status':'failed'}
json.dump(summary,open(E/'metrics.json','w'),indent=2);print(json.dumps(summary,indent=2))
