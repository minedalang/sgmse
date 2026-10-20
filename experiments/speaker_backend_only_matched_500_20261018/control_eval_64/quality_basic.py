import csv,json,time
from pathlib import Path
import numpy as np,soundfile as sf
from pesq import pesq
from pystoi import stoi
R=Path('/data/coding/sgmse'); M=R/'experiments/speaker_headroom_replication_64_20261017/manifest.csv'; C=R/'experiments/speaker_backend_only_matched_500_20261018/control_eval_64/control'; O=R/'experiment_logs/speaker_headroom_replication_64_20261017/off'; G=R/'experiment_logs/speaker_headroom_replication_64_20261017/router'
def sisdr(s,e):
 s=s-s.mean();e=e-e.mean();a=np.dot(e,s)/(np.dot(s,s)+1e-12);t=a*s;return 10*np.log10((np.dot(t,t)+1e-12)/(np.dot(e-t,e-t)+1e-12))
def load(p):return sf.read(p,dtype='float32',always_2d=False)[0]
rows=list(csv.DictReader(M.open())); vals={a:{m:[] for m in ('pesq','stoi','sisdr')} for a in ('control','off','router')}
for i,r in enumerate(rows):
 s=load(Path(r['clean_path']))
 for a,p in [('control',C/r['filename']),('off',O/r['filename']),('router',G/r['filename'])]:
  e=load(p); vals[a]['pesq'].append(float(pesq(16000,s,e,'wb')));vals[a]['stoi'].append(float(stoi(s,e,16000,extended=False)));vals[a]['sisdr'].append(float(sisdr(s,e)))
out={'n':len(rows),'metrics':{}}
for m in ('pesq','stoi','sisdr'):
 out['metrics'][m]={a:float(np.mean(v[m])) for a,v in vals.items()}; out['metrics'][m]['control_minus_off']=float(np.mean(vals['control'][m])-np.mean(vals['off'][m]));out['metrics'][m]['control_minus_router']=float(np.mean(vals['control'][m])-np.mean(vals['router'][m]))
json.dump(out,open(C.parent/'quality_basic_metrics.json','w'),indent=2);print(json.dumps(out,indent=2))
