"""Synthetic endpoints only; no actual corpus samples or model imports."""
from pathlib import Path
import importlib.util,json,math,warnings,tempfile,sys,types
from helpers import module,aggregate,sha
from headroom import analyze
from quality import sisdr,metric,measure_all
from independent_endpoints import verify_headroom,verify_quality
STAGE=Path(__file__).resolve().parent
cal=module('synthetic_score',STAGE.parent.parent/'p8_voicebank_clean_text_calibration_20261019/run_calibration.py')
cases=[];items=[]
for i in range(8):
 for n in ['N1','N2']:
  for q in [-18,0]:
   name=f'fake{i}_{n}_{q}.wav';r=dict(basename=name,id='fake'+str(i),speaker='speaker'+str(i//2),noise=n,snr_db=q);cases.append(r)
   for arm in ['noisy','frontend','off']:
    hyp='a b c' if (i%2==0 and arm=='frontend') or (i%2 and arm=='off') else 'a z'
    items.append(dict(key=name+'__'+arm,id=r['id'],arm=arm,score=cal.score('a b c',hyp)))
reg=dict(cases=cases,groups=[dict(id='fake'+str(i),scaledclean=dict(wav='fake'+str(i)+'.wav')) for i in range(8)]);
from run_pilot import asr_requests
from collections import Counter
for r in cases:r['wav']=r['basename']
mock_native=[dict(basename=r['basename'],frontend_pcm=dict(path='frontend/'+r['basename']),off_pcm=dict(path='off/'+r['basename'])) for r in cases]
req=asr_requests(reg,mock_native,Path('/mock/input'),Path('/mock/model'))
assert Counter(r['arm'] for r in req)==dict(scaledclean=8,noisy=32,frontend=32,off=32)
d=dict(items=items,headroom=analyze(items,cases,aggregate))
assert d['headroom']['strata']['all']['headroom']>0
assert verify_headroom(d,reg,aggregate)['bootstrap_replicates']==10000
# Deterministic ties and undefined reference denominators.
fake=cal.score('','hi');assert fake['word']['rate'] is None and aggregate([fake])['word']['micro_rate'] is None
assert sisdr([0,0],[0,1])[0] is None
assert sisdr([-1,1],[-2,2])==(None,'perfect residual')
assert math.isfinite(sisdr([0,1,0,-1],[.1,.8,-.1,-.9])[0])
def sentinel():warnings.warn('Not enough STFT frames',RuntimeWarning);return 1e-5
assert metric(sentinel,True)['value'] is None and metric(sentinel,True)['raw_value']==1e-5
assert metric(lambda:float('nan'))['value'] is None and metric(lambda:float('nan'))['raw_value_repr']=='nan'
json.dumps(metric(lambda:float('inf')),allow_nan=False)
json.dumps(metric(lambda:float('nan')),allow_nan=False)
assert metric(lambda:1/0)['error'] is not None
# Real libsndfile only generated WAVs; fake metric routines exercise errors and paired-null bookkeeping.
import numpy as np,soundfile as sf
fake_stoi=types.ModuleType('pystoi');fake_pesq=types.ModuleType('pesq')
def stoi(ref,x,fs_sig,extended=False):
 assert fs_sig==16000 and not extended
 if float(x[0])<0:return sentinel()
 return .7
fake_stoi.stoi=stoi
fake_pesq.pesq=lambda fs,ref,x,mode='wb':2.5
before={k:sys.modules.get(k) for k in ['pystoi','pesq']};sys.modules.update(pystoi=fake_stoi,pesq=fake_pesq)
try:
 with tempfile.TemporaryDirectory() as temp:
  root=Path(temp);inp=root/'input';out=root/'out';inp.mkdir();out.mkdir();reg['groups']=[];native=[]
  y=np.sin(np.arange(640)*.11)*.2;y[0]=.02
  for i in range(8):
   name=f'fake{i}_clean.wav';sf.write(inp/name,y,16000,subtype='PCM_16');reg['groups'].append(dict(id='fake'+str(i),scaledclean=dict(wav=name)))
  for i,r in enumerate(cases):
   r['wav']=r['basename'];sf.write(inp/r['wav'],y+.01,16000,subtype='PCM_16');n=dict(basename=r['basename'])
   for arm in ['frontend','off']:
    x=y*.9+np.cos(np.arange(640)*.17)*.01;x[0]=-.02 if i%3==0 and arm=='off' else .03
    path=r['basename']+'_'+arm+'.wav';sf.write(out/path,x,16000,subtype='PCM_16');n[arm+'_pcm']=dict(path=path)
   native.append(n)
  d['native_cases']=native;d['quality']=measure_all(reg,native,inp,out,lambda:None)
  result=verify_quality(d,reg,inp,out);assert result['cases']==32
  assert d['quality']['strata']['all']['stoi']['off_minus_frontend_paired']['paired_n']==21
finally:
 for k,v in before.items():
  if v is None:sys.modules.pop(k,None)
  else:sys.modules[k]=v
result=dict(status='passed',synthetic_native_cases=32,synthetic_ASR_requests=104,synthetic_quality_arms=96,independent_headroom_bootstrap=10000,strata=21,zero_N_and_ties=True,SI_SDR_degeneracy=True,warning_raw_null=True,paired_missing_case_check=True,real_source_audio_decodes=0,model_imports=0)
(STAGE/'endpoint_test_results.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
