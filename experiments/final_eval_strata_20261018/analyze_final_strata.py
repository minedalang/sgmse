import csv,re,json,statistics,math
from pathlib import Path
root=Path('/data/coding/sgmse/experiments/final_eval_full_20261018')
mani=Path('/data/coding/sgmse/experiments/final_eval_manifest_20261018/manifest.csv')
rows={}
with mani.open() as f:
 for r in csv.DictReader(f):
  m=re.search(r'_snr(-?\d+)(?:_|\.)',r['noisy_path'])
  if not m: raise ValueError(r['noisy_path'])
  rows[(r['split'],r['file_id'])]={'split':r['split'],'file_id':r['file_id'],'snr':int(m.group(1))}
metrics=['PESQ_e','STOI_e','SI-SDR_e','DNSMOS_SIG_e','DNSMOS_OVRL_e','DNSMOS_P808_e','DNSMOS_BAK_e']
for arm in ['off','router','tcut005']:
 for split in ['no_reverb','with_reverb']:
  p=root/'metrics'/arm/f'{split}.csv'
  with p.open() as f:
   for r in csv.DictReader(f):
    fid=Path(r['utt']).stem.replace('clean_','')
    # CSV uses clean_fileid_N; manifest uses fileid_N
    k=(split,fid)
    if k not in rows:
     raise ValueError((k,r['utt']))
    rows[k][arm]={m:float(r[m]) for m in metrics}
# add deltas and stratify
for k,r in rows.items():
 if any(a not in r for a in ['off','router','tcut005']): raise ValueError(k)
 for arm in ['router','tcut005']:
  r[arm+'_delta']={m:r[arm][m]-r['off'][m] for m in metrics}

def mean(xs): return sum(xs)/len(xs) if xs else float('nan')
def ci_boot(xs, B=10000, seed=20261018):
 import random
 rng=random.Random(seed)
 n=len(xs); vals=[]
 for _ in range(B): vals.append(mean([xs[rng.randrange(n)] for _ in range(n)]))
 vals.sort(); return [vals[int(.025*B)],vals[int(.975*B)]]
def summarize(sub, arm):
 out={'n':len(sub),'snr_values':sorted(set(r['snr'] for r in sub)),'metrics':{}}
 for m in metrics:
  x=[r[arm+'_delta'][m] for r in sub]
  out['metrics'][m]={'mean':mean(x),'ci95':ci_boot(x),'n_positive':sum(v>0 for v in x),'n_negative':sum(v<0 for v in x)}
 return out
bins=[('snr<=6',lambda s:s<=6),('snr7-12',lambda s:7<=s<=12),('snr>=13',lambda s:s>=13)]
out={'experiment_id':'final_eval_strata_20261018','source':'final_eval_full_20261018','metrics':metrics,'seed':20261018,'bins':{}}
for split in ['no_reverb','with_reverb']:
 sub=[r for (sp,_),r in rows.items() if sp==split]
 out['bins'][split]={'all':{a:summarize(sub,a) for a in ['router','tcut005']}}
 for name,fn in bins:
  ss=[r for r in sub if fn(r['snr'])]
  out['bins'][split][name]={a:summarize(ss,a) for a in ['router','tcut005']}
# logistic simple relationship not inferential: correlations with snr
out['descriptive_snr_correlations']={}
for split in ['no_reverb','with_reverb']:
 sub=[r for (sp,_),r in rows.items() if sp==split]
 for arm in ['router','tcut005']:
  out['descriptive_snr_correlations'][split+'_'+arm]={}
  for m in metrics:
   xs=[r['snr'] for r in sub]; ys=[r[arm+'_delta'][m] for r in sub]
   mx,my=mean(xs),mean(ys); den=(sum((x-mx)**2 for x in xs)*sum((y-my)**2 for y in ys))**.5
   out['descriptive_snr_correlations'][split+'_'+arm][m]=sum((x-mx)*(y-my) for x,y in zip(xs,ys))/den if den else None
out['rows']=[r for r in rows.values()]
Path('/data/coding/sgmse/experiments/final_eval_strata_20261018').mkdir(exist_ok=True)
json.dump(out,open('/data/coding/sgmse/experiments/final_eval_strata_20261018/metrics.json','w'),indent=2)
# compact csv
with open('/data/coding/sgmse/experiments/final_eval_strata_20261018/per_utterance.csv','w',newline='') as f:
 fields=['split','file_id','snr']+[f'{a}_{m}' for a in ['router','tcut005'] for m in metrics]
 w=csv.DictWriter(f,fieldnames=fields);w.writeheader()
 for r in rows.values():
  z={x:r[x] for x in ['split','file_id','snr']}
  for a in ['router','tcut005']:
   for m in metrics:z[f'{a}_{m}']=r[a+'_delta'][m]
  w.writerow(z)
