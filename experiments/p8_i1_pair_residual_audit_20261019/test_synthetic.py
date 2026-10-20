"""Synthetic arrays/temp WAV only. Never opens real I0 audio."""
from pathlib import Path
import ast,hashlib,importlib.util,json,math,os,struct,subprocess,sys,tempfile,time,wave
for n in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS']:os.environ[n]='1'
import numpy as np
OUT=Path(__file__).resolve().parent

def module(name,file):
 spec=importlib.util.spec_from_file_location(name,OUT/file);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
a=module('business','audit_pairs.py');v=module('independent','validate_audit.py');checks=[]
def ok(x,label):assert x,label;checks.append(label)
def compare_case(s,x,label):
 r,l,f=a.compute(s,x,'synthetic');rr,ll,ff=v.independent_compute(s,x,'synthetic');v.compare(r,rr);v.compare(l,ll,abs_=1e-10);v.compare(f,ff);json.dumps(r,allow_nan=False);checks.append(label);return r,l,f
rng=np.random.default_rng(20261019);s=rng.normal(0,.1,1603)
for lag in [-160,-7,0,11,160]:
 x=np.zeros_like(s)
 if lag>=0:x[lag:]=s[:len(s)-lag]
 else:x[:lag]=s[-lag:]
 r,_,_=compare_case(s,x,'known lag '+str(lag));ok(r['best_lag']['lag']==lag,'lag sign/boundary '+str(lag))
r,_,f=compare_case(s,1.3*s+.03,'gain non1 plus DC');ok(abs(r['full_zero_lag_OLS']['ols_gain']-1)>0,'gain recorded no normalization');ok(f[-1]['n']==3 and len(f)==11,'partial tail every hop')
r,lr,fr=compare_case(np.ones(803)*.1,np.ones(803)*.2,'constant centered denominator')
ok(r['full_zero_lag_OLS']['pearson'] is None and r['best_lag'] is None and all(row['pearson'] is None for row in lr),'non-dyadic mathematical constant explicitly Pearsonnull')
compare_case(np.zeros(803),np.zeros(803),'all zero null JSON')
compare_case(s,-s,'negative correlation retained')
rows=[dict(lag=-3,pearson=.8),dict(lag=3,pearson=.8+5e-13),dict(lag=5,pearson=.7)]
ok(a.best(rows)['lag']==v.independent_best(rows)['lag']==-3,'absolute tolerance tie(abs lag,lag)')
ok(a.best([dict(lag=0,pearson=None)]) is None,'all undefined bestnull')
x=s*.8+.04;rr=a.relation(s,x)
ok(abs(rr['fit_residual_s_inner_product'])<1e-12 and abs(rr['fit_residual_pearson'])>.01,'OLS inner orthogonality not centered independence')
for bad in [np.full(10,np.nan),np.full(10,np.inf)]:
 arow=a.relation(bad,np.ones(10));brow=v.independent_relation(bad,np.ones(10));v.compare(arow,brow);json.dumps(arow,allow_nan=False);ok(arow['pearson'] is None and arow['numeric_reason']=='nonfinite samples','nonfinite null and reason')
with tempfile.TemporaryDirectory(prefix='synthetic_',dir=OUT) as tmp:
 root=Path(tmp);p=root/'fake.wav'
 with wave.open(str(p),'wb') as w:w.setparams((1,2,16000,0,'NONE',''));w.writeframes(struct.pack('<hhhh',0,1,-32768,32767))
 lock=dict(bytes=p.stat().st_size,sha256=a.sha(p),header=dict(frames=4));counts=dict(pcm_file_decodes=0,pcm_samples_decoded=0,pcm_payload_bytes_decoded=0,full_wav_hash_bytes=0)
 decoded=a.decode_locked_pcm(p,lock,counts);ok(decoded.tolist()==[0,1/32768,-1,32767/32768] and counts['pcm_file_decodes']==1,'synthetic PCM endian endpoints and counters')
 p.write_bytes(p.read_bytes()[:-1]);truncated=dict(bytes=p.stat().st_size,sha256=a.sha(p),header=dict(frames=4))
 try:a.decode_locked_pcm(p,truncated,counts)
 except RuntimeError:checks.append('truncated synthetic PCM rejected')
 else:raise AssertionError('truncated accepted')
 from unittest.mock import patch
 # Only override path anchoring for generated WAV container; real guard tested independently below.
 with patch.object(Path,'resolve',return_value=Path('/escape/fake.wav')):
  try:a.decode(p,lock,counts)
  except RuntimeError:checks.append('resolved path escape rejected')
  else:raise AssertionError('escape accepted')
 bad={str(p):dict(bytes=p.stat().st_size,sha256='0'*64)}
 try:a.verify(bad)
 except RuntimeError:checks.append('wrong SHA rejected')
 else:raise AssertionError('wrong SHA accepted')
 # Main begin must reject existing marker without importing real sources.
 old=a.OUT;a.OUT=root;(root/'STARTED.json').write_text('{}')
 cfg=dict(budgets=dict(main_seconds=120,validation_seconds=120,artifact_bytes=8388608))
 try:a.begin('main',cfg)
 except RuntimeError:checks.append('existing main marker rejected')
 else:raise AssertionError('duplicate accepted')
 (root/'STARTED.json').unlink()
 beginstart,finish,check=a.begin('main',cfg)
 import signal
 try:
  with patch.object(time,'monotonic',return_value=beginstart+119):
   try:check()
   except RuntimeError:checks.append('118s work deadline guard')
   else:raise AssertionError('timeout accepted')
  with patch.object(a,'size',return_value=8388608):
   try:check()
   except RuntimeError:checks.append('8MiB artifact reserve guard')
   else:raise AssertionError('bytes accepted')
  finish();ok(signal.getitimer(signal.ITIMER_REAL)[0]>0,'absolute finalization alarm stays armed')
 finally:signal.setitimer(signal.ITIMER_REAL,0)
 a.OUT=old
 pre="import importlib.util;from pathlib import Path;s=importlib.util.spec_from_file_location('a',"+repr(str(OUT/'audit_pairs.py'))+");a=importlib.util.module_from_spec(s);s.loader.exec_module(a);e=[];a.guard(Path("+repr(str(root))+"),set(),e);"
 for action in ['import torch','import socket;socket.socket()',"open('/data/i1_forbidden','w')","open('/data/datasets/forbidden.wav','rb')"]:
  p=subprocess.run([sys.executable,'-B','-c',pre+action],capture_output=True,text=True);ok(p.returncode!=0,'guard '+action)
 # Test recorded caught network event still visible, without real socket creation.
 action="\ntry:\n import socket;socket.socket()\nexcept RuntimeError:pass\nassert len(e)==1 and e[0]['kind']=='forbidden_network_process'"
 p=subprocess.run([sys.executable,'-B','-c',pre+action],capture_output=True,text=True);ok(p.returncode==0,'caught forbidden network attempt recorded')
for p in OUT.glob('*.py'):ast.parse(p.read_text())
ok(not any((OUT/n).exists() for n in ['STARTED.json','VALIDATION_STARTED.json','metrics.json']),'no real execution artifacts')
result=dict(status='passed',count=len(checks),checks=checks,real_pcm_file_decodes=0,real_header_reads=0,models=0,network_bytes=0)
(OUT/'synthetic_test_results.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
