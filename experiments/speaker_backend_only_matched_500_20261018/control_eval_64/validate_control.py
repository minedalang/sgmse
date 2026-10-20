import csv,hashlib,json
from pathlib import Path
import soundfile as sf,numpy as np
root=Path('/data/coding/sgmse'); m=root/'experiments/speaker_headroom_replication_64_20261017/manifest.csv'; out=root/'experiments/speaker_backend_only_matched_500_20261018/control_eval_64/control'
def h(p):
 x=hashlib.sha256();x.update(p.read_bytes());return x.hexdigest()
rows=list(csv.DictReader(m.open())); rec=[]
for r in rows:
 p=out/r['filename']; x,s=sf.read(p,dtype='float32',always_2d=False); assert s==16000 and x.ndim==1 and len(x)==int(r['samples']) and np.isfinite(x).all(); assert h(Path(r['noisy_path']))==r['noisy_sha256'] and h(Path(r['clean_path']))==r['clean_sha256']; rec.append({'filename':r['filename'],'control_sha256':h(p),'subtype':sf.info(p).subtype})
json.dump({'n':len(rec),'all_finite_aligned':True,'subtype':'PCM_16','manifest_sha256':h(m)},open(out.parent/'control_output_validation.json','w'),indent=2)
