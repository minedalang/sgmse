import csv,json,time,traceback,hashlib
from pathlib import Path
import soundfile as sf
import sys
sys.path.insert(0,'/data/coding/mymetrics'); import get_metrics_for_papertable_forDNS as dm
E=Path('/data/coding/sgmse/experiments/rrsd_harm_oracle_n30_4_dnsmos_20261019'); R=Path('/data/coding/sgmse'); arms={'full':R/'experiment_logs/rrsd_harm_oracle_n30_4_full','gate':R/'experiment_logs/rrsd_harm_oracle_n30_4_gate'}
files=sorted(p.name for p in arms['full'].glob('*.wav'))[:4]; rows=[]; t=time.time()
for f in files:
 z={'filename':f}
 for a,d in arms.items():
  p=d/f; x,sr=sf.read(p,dtype='float32'); z[f'{a}_path']=str(p); z[f'{a}_sha256']=hashlib.sha256(p.read_bytes()).hexdigest(); z[f'{a}_frames']=len(x); z[f'{a}_sr']=sr
  try:z[a]=dm.calc_dnsmos(x,sr=sr,device='auto')
  except Exception as e:z[a+'_error']=repr(e)
 rows.append(z)
(E/'paired_dnsmos.csv').write_text('')
keys=['filename']+[f'{a}_{k}' for a in arms for k in ('p808','sig','bak','ovrl')]
with (E/'paired_dnsmos.csv').open('w',newline='') as h:
 w=csv.DictWriter(h,fieldnames=keys); w.writeheader()
 for z in rows:
  w.writerow({k: z.get(k) if '_' in k else z.get(k) for k in keys})
import numpy as np
summary={}
for k in ('p808','sig','bak','ovrl'):
 full=np.array([r['full'][k] for r in rows]); gate=np.array([r['gate'][k] for r in rows]); d=gate-full
 rng=np.random.default_rng(20261019); b=np.mean(d[rng.integers(0,4,(10000,4))],axis=1)
 summary[k]={'full_mean':float(full.mean()),'gate_mean':float(gate.mean()),'gate_minus_full_mean':float(d.mean()),'bootstrap_95ci':[float(np.quantile(b,.025)),float(np.quantile(b,.975))],'per_utterance_delta':d.tolist()}
out={'experiment_id':'rrsd_harm_oracle_n30_4_dnsmos_20261019','n':4,'metrics':summary,'bootstrap':{'replicates':10000,'seed':20261019,'unit':'utterance (exploratory)'},'runtime_seconds':time.time()-t,'device_cache':{str(k):str(v) for k,v in dm._DNSMOS_DEVICE_CACHE.items()}}
(E/'metrics.json').write_text(json.dumps(out,indent=2)+'\n'); print(json.dumps(out,indent=2))
