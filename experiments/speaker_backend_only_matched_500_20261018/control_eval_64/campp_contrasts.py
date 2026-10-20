import csv,json
from pathlib import Path
import numpy as np
E=Path(__file__).parent; rows=list(csv.DictReader((E/'per_utterance_campp.csv').open())); sp=sorted({r['speaker'] for r in rows}); idx={s:[i for i,r in enumerate(rows) if r['speaker']==s] for s in sp}; rng=np.random.default_rng(20261017); draws=rng.integers(0,len(sp),size=(5000,len(sp))); C=np.array([np.bincount(x,minlength=len(sp)) for x in draws]); out={}
for metric in ['paired_clean_cosine']:
 for b,a in [('control','off'),('control','router')]:
  d=np.array([float(r[f'{b}_{metric}'])-float(r[f'{a}_{metric}']) for r in rows]); sd=np.array([d[idx[s]].mean() for s in sp]); boot=(C*sd).sum(1)/C.sum(1); out[f'{b}_minus_{a}_{metric}']={'mean_delta':float(d.mean()),'speaker_cluster_bootstrap_95ci':[float(np.quantile(boot,.025)),float(np.quantile(boot,.975))]}
(E/'campp_contrasts.json').write_text(json.dumps({'bootstrap':{'replicates':5000,'seed':20261017,'unit':'speaker cluster'},'contrasts':out},indent=2)+'\n'); print(json.dumps(out,indent=2))
