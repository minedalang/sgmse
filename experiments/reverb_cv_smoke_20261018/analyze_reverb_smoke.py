import csv,json,random
from pathlib import Path
exp=Path('/data/coding/sgmse/experiments/reverb_cv_smoke_20261018')
metrics=['PESQ_e','STOI_e','SI-SDR_e']
D={}
for arm in ['off','router']:
 for cond in ['identity','short','medium']:
  with (exp/'metrics'/arm/f'{cond}.csv').open() as f:
   for row in csv.DictReader(f):
    u=Path(row['utt']).stem
    D.setdefault(u,{}).setdefault(cond,{})[arm]={m:float(row[m]) for m in metrics}
assert len(D)==4

def mean(x):return sum(x)/len(x)
def ci(x,B=10000):
 rng=random.Random(20261018); n=len(x); ys=sorted(mean([x[rng.randrange(n)] for _ in range(n)]) for _ in range(B));return [ys[int(.025*B)],ys[int(.975*B)]]
out={'experiment_id':'reverb_cv_smoke_20261018','n_utterances':len(D),'metrics':metrics,'bootstrap_seed':20261018,'bootstrap_replicates':10000,'bootstrap_unit':'utterance','results':{}}
for c in ['identity','short','medium']:
 out['results'][c]={'router_minus_off':{}}
 for m in metrics:
  x=[D[u][c]['router'][m]-D[u][c]['off'][m] for u in D]
  out['results'][c]['router_minus_off'][m]={'mean':mean(x),'ci95':ci(x),'values':x}
for c in ['short','medium']:
 out['results'][c]['interaction_vs_identity']={}
 for m in metrics:
  x=[(D[u][c]['router'][m]-D[u][c]['off'][m])-(D[u]['identity']['router'][m]-D[u]['identity']['off'][m]) for u in D]
  out['results'][c]['interaction_vs_identity'][m]={'mean':mean(x),'ci95':ci(x),'values':x}
json.dump(out,open(exp/'metrics.json','w'),indent=2)
print(json.dumps(out,indent=2))
