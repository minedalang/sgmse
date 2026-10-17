import csv,json,random,math
from pathlib import Path
root=Path('/data/coding/sgmse/experiments/final_eval_full_20261018')
metrics=['PESQ_e','STOI_e','SI-SDR_e','DNSMOS_SIG_e','DNSMOS_OVRL_e','DNSMOS_P808_e','DNSMOS_BAK_e']
D={}
for arm in ['off','router','tcut005']:
 for split in ['no_reverb','with_reverb']:
  with (root/'metrics'/arm/f'{split}.csv').open() as f:
   for r in csv.DictReader(f):
    fid=Path(r['utt']).stem.replace('clean_','')
    D.setdefault(fid,{}).setdefault(arm,{})[split]={m:float(r[m]) for m in metrics}
ids=sorted(D)
assert len(ids)==150 and all(set(D[i][a])=={'no_reverb','with_reverb'} for i in ids for a in ['off','router','tcut005'])
def mean(x):return sum(x)/len(x)
def ci(x,B=10000):
 rng=random.Random(20261018); n=len(x); y=sorted(mean([x[rng.randrange(n)] for _ in range(n)]) for _ in range(B)); return [y[int(.025*B)],y[int(.975*B)]]
out={'experiment_id':'final_eval_reverb_interaction_20261018','n':len(ids),'metrics':metrics,'bootstrap_seed':20261018,'bootstrap_replicates':10000,'unit':'matched_file_id'}
for arm in ['off','router','tcut005']:
 out[arm]={}
 for m in metrics:
  # reverb effect in arm
  vals=[D[i][arm]['with_reverb'][m]-D[i][arm]['no_reverb'][m] for i in ids]
  out[arm][m]={'reverb_minus_no_reverb_mean':mean(vals),'ci95':ci(vals)}
for arm in ['router','tcut005']:
 out[arm+'_interaction_vs_off']={}
 for m in metrics:
  # effect of reverb on arm-off delta: (arm-off)_with - (arm-off)_no
  vals=[(D[i][arm]['with_reverb'][m]-D[i]['off']['with_reverb'][m])-(D[i][arm]['no_reverb'][m]-D[i]['off']['no_reverb'][m]) for i in ids]
  out[arm+'_interaction_vs_off'][m]={'mean':mean(vals),'ci95':ci(vals),'n_positive':sum(v>0 for v in vals),'n_negative':sum(v<0 for v in vals)}
json.dump(out,open('/data/coding/sgmse/experiments/final_eval_reverb_interaction_20261018_metrics.json','w'),indent=2)
