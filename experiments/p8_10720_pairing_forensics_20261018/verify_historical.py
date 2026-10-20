"""Supplement historical routes and exact PCM origin; writes new artifact only."""
from pathlib import Path
import json,io,urllib.request,hashlib
import soundfile as sf
import numpy as np
OUT=Path(__file__).resolve().parent
if (OUT/'historical_metrics.json').exists():raise RuntimeError('Refuse overwrite')
d=json.loads((OUT/'metrics.json').read_text());E=OUT.parent;result={'status':'completed','initial_execution_failure':{'command':'python verify.py','error':'ModuleNotFoundError: soundfile','resolution':'Used existing /data/coding/sgmse/env/bin/python, no installation'},'arms':{}}
for arm,inf in d['sources'].items():
 p=E/'p8_e1_candidate_listening_20261018/clips'/f'fileid_10720_14.50_15.00_{arm}.wav';b=p.read_bytes();a,sr=sf.read(io.BytesIO(b));s,_=sf.read(inf['path']);start=4.75 if arm=='noisy' else 13.75; expected=s[round(start*sr):round((start+2)*sr)];url='http://127.0.0.1:8766/p8_e1_candidate_listening_20261018/clips/'+p.name
 with urllib.request.urlopen(url,timeout=10) as r:served=r.read();status=r.status
 result['arms'][arm]=dict(path=str(p),url=url,http_status=status,http_matches_disk=served==b,sha256=hashlib.sha256(b).hexdigest(),sample_rate=sr,samples=len(a),verified_source_window_s=[start,start+2],exact_pcm_equality=bool(np.array_equal(a,expected)),equals_current_clip=bool(np.array_equal(a,sf.read(d['clips'][arm]['path'])[0])))
(OUT/'historical_metrics.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
