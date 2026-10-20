"""Generated arrays and temporary container bytes only, no corpus IO or true run."""
from pathlib import Path
import ast,hashlib,io,json,math,os,subprocess,sys,tempfile,time,wave
for k in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS']:os.environ[k]='1'
import numpy as np
import event_integrity as main
from independent_integrity import independent_pair,compare
import runtime
OUT=Path(__file__).resolve().parent;checks=[]
def ok(x,name):assert x,name;checks.append(name)
def container(y):
    b=io.BytesIO()
    with wave.open(b,'wb') as w:w.setparams((1,2,16000,0,'NONE',''));w.writeframes(np.rint(y*32768).astype('<i2').tobytes())
    return b.getvalue()
def pairtest(s,x,name):
    cb=container(s);xb=container(x);a=main.pair(s,x,cb,xb);b=independent_pair(s,x,cb,xb);compare(a,b);json.dumps(a,allow_nan=False);ok(len(a['events'])==2 and len(a['operations'])==8,name);return a
# SmallFFT independent direct complex DFT (no FFT business reuse).
rng=np.random.default_rng(20261019);v=rng.normal(size=16)
direct=np.asarray([sum(v[n]*complex(math.cos(-2*math.pi*k*n/16),math.sin(-2*math.pi*k*n/16)) for n in range(16)) for k in range(9)])
ok(np.allclose(direct,np.fft.rfft(v),rtol=1e-12,atol=1e-12),'smallFFT directDFT reference')
# Fixed512 geometry crosscheck on all high bins via independent complex matrix.
z=rng.normal(size=320);h=.5-.5*np.cos(2*np.pi*np.arange(320)/319)
dft=np.exp(-2j*np.pi*np.arange(257)[:,None]*np.arange(320)[None,:]/512)@(z*h)
ok(np.allclose(dft,np.fft.rfft(z*h,512),rtol=1e-10,atol=1e-10),'512 symmetricHann directDFT frequency geometry')
s=np.zeros(19200);s[4800:5120]=.1*np.sin(2*np.pi*6000*np.arange(320)/16000);s[11040:13600]=.08*np.sin(2*np.pi*4000*np.arange(2560)/16000);x=s+rng.normal(0,.01,len(s));result=pairtest(s,x,'knownHF transient sustained complete independent reconstruction')
ok(all(e['status']=='available' for e in result['events']),'positive fixed candidates')
ok(result['events'][0]['start']+1440<=result['events'][1]['start'] or result['events'][1]['start']+1440<=result['events'][0]['start'],'target intervals plus160gap')
ok(any(r.get('clean_delta_energy_difference_vs_target') not in [None,0.] for r in result['operations'] if r['site']=='position_control'),'actual tapered control energy mismatch reported')
ok(len(result['frames'])==len(range(0,len(s)-319,160)) and len(result['candidates']['transient'])>1,'allframe and candidate rows retained')
pairtest(np.zeros(10000),np.zeros(10000),'zero allunavailable eight operations retained')
constant=pairtest(np.ones(10000)*.1,np.ones(10000)*.12,'constantHann leakage described not eventtruth')
ok(all(r['F']==0 for r in constant['frames'][1:]),'constant identical normalized spectra flux0')
noctrl=pairtest(np.ones(9000)*.1,np.ones(9000)*.12,'valid sustained no control retain unavailable')
ok(noctrl['events'][1]['status']=='available' and noctrl['events'][1]['position_control_start'] is None,'missing control not relaxed')
short=pairtest(np.zeros(7000),np.zeros(7000),'boundary no eligible window unavailable')
ok(not short['candidates']['transient'] and not short['candidates']['sustained'],'margins not relaxed')
rows=[dict(start=80,score=.5),dict(start=40,score=.5-5e-13),dict(start=0,score=.49)]
ok(main.choose(rows,'score')['start']==40,'within tolerance earliest scoretie')
ok(main.separated(0,1440) and not main.separated(0,1439),'gap equality and overlap')
for family in ['cosine','linear']:
 m=main.mask(family);ok(len(m)==1280 and m[0]==m[-1]==0 and np.array_equal(m,m[::-1]) and np.all(m[160:1120]==1),'mask '+family+' endpoints mirror plateau')
 # source has varying envelope so tapered energies differ by position despite same untapered control rule.
e=dict(event='synthetic',start=3200,position_control_start=6400);y=rng.normal(0,.1,10000);r=main.operation(y,y+.01,e,'cosine','target');ok(r['maskoutside_clean_equal'] and r['maskoutside_noisy_equal'],'maskoutside source exact');ok(r['delta_energy']>0 and r['float_residual_reconstruction_max_error']<1e-12,'delta and rawresidual algebra only')
zero=main.operation(np.zeros(10000),np.ones(10000)*.1,e,'linear','target');ok(zero['delta_energy']==0 and zero['applied_delta_max_error']==0,'delta0 operation')
q=main.quantize(np.asarray([.5/32768,1.5/32768,-.5/32768,-1.5/32768]));expected=np.asarray([0,2,0,-2],dtype='<i2').tobytes();ok(q['pcm_payload_sha256']==hashlib.sha256(expected).hexdigest(),'ties-even negative halfLSB')
near=main.quantize(np.asarray([32767/32768+1e-9]));ok(near['status']=='unsafe' and near['float_out_of_range']==1 and near['rounded_out_of_range']==0 and near['pcm_payload_sha256'] is None,'floatunsafe although rounded safe no clipping')
far=main.quantize(np.asarray([1.]));ok(far['status']=='unsafe' and far['rounded_out_of_range']==1,'roundunsafe retained no payload')
nonfinite=main.quantize(np.asarray([np.nan]));json.dumps(nonfinite,allow_nan=False);ok(nonfinite['status']=='unsafe','nonfinite null JSON')
# Edited X can be unsafe while source was valid: subtracting positiveS from negativeX.
unsafe=main.operation(np.ones(10000)*.9,np.ones(10000)*-.9,e,'linear','target');ok(unsafe['status']=='unsafe' and unsafe['noisy_edited']['pcm_payload_sha256'] is None,'unsafe edited arm remains slot')
for role in ['clean','noisy']:ok(result['original_sham'][role]['byte_identical'] and result['original_sham'][role]['original_full_wav_sha256']==result['original_sham'][role]['identity_sham_full_wav_sha256'],'original sham fullbyte identity '+role)
bytes_=container(np.asarray([0,1/32768,-1,32767/32768]));aa,pp=runtime.decode_container(bytes_,4);bb,_=runtime.decode_container(bytes_,4,True);ok(np.array_equal(aa,bb) and len(pp)==8,'synthetic originalPCM independent roundtrip')
for damaged,frames,label in [(bytes_[:-1],4,'truncated header payload'),(bytes_,5,'header framecount mismatch')]:
 try:runtime.decode_container(damaged,frames)
 except (RuntimeError,EOFError,wave.Error):checks.append(label)
 else:raise AssertionError('bad container accepted')
with tempfile.TemporaryDirectory(prefix='synthetic_',dir=OUT) as tmp:
 root=Path(tmp);p=root/'source.txt';p.write_text('synthetic');bad={str(p):dict(bytes=p.stat().st_size,sha256='0'*64)}
 try:runtime.verify(bad)
 except RuntimeError:checks.append('sourcehash mismatch')
 else:raise AssertionError('badhash accepted')
 from unittest.mock import patch
 count=dict(pcm_file_decodes=0,pcm_samples_decoded=0,pcm_payload_bytes_decoded=0,decode_precheck_wav_hash_bytes=0,identity_original_wav_bytes_read=0)
 try:runtime.decode(root/'bad.wav',dict(),count)
 except RuntimeError:checks.append('sourcepath rejected before read')
 else:raise AssertionError('path accepted')
 old=runtime.ROOT;runtime.ROOT=root;(root/'STARTED.json').write_text('{}')
 try:runtime.start('main',dict(),[])
 except RuntimeError:checks.append('existing marker noretry')
 else:raise AssertionError('marker accepted')
 runtime.ROOT=old
 # fresh child installs single-process deadline/byte/model/network/write guards using synthetic root only.
 pre="import sys, pathlib;sys.path.insert(0,"+repr(str(OUT))+");import runtime;r=pathlib.Path("+repr(str(root))+');runtime.ROOT=r;e=[];c={"inherited_audio_locks":{},"budgets":{"artifact_bytes":8388608}};'
 (root/'STARTED.json').unlink()
 actions=[("runtime.time.monotonic=lambda:b+119;check()",'118s workdeadline'),("runtime.size=lambda:8388608;check()",'8MiB artifact reserve'),('import torch','model forbidden'),('import socket;socket.socket()','network forbidden'),("open('/data/i1b_forbidden','w')",'sourcewrite forbidden'),("open(str(r/'new.wav'),'wb')",'newwaveform export forbidden')]
 for index,(action,label) in enumerate(actions):
  for f in ['STARTED.json','VALIDATION_STARTED.json']:
   if (root/f).exists():(root/f).unlink()
  child=subprocess.run([sys.executable,'-B','-c',pre+"b,check,finalize,finish=runtime.start('main',c,e);"+action],capture_output=True,text=True);ok(child.returncode!=0,label)
 for f in ['STARTED.json','VALIDATION_STARTED.json']:
  if (root/f).exists():(root/f).unlink()
 child=subprocess.run([sys.executable,'-B','-c',pre+"b,check,finalize,finish=runtime.start('main',c,e)\ntry:\n import socket;socket.socket()\nexcept RuntimeError:pass\nassert e\ncheck()"],capture_output=True,text=True);ok(child.returncode!=0 and 'Forbidden attempted' in child.stderr,'caught forbidden attempt fails checkpoint')
for f in OUT.glob('*.py'):ast.parse(f.read_text())
ok(not any((OUT/f).exists() for f in ['STARTED.json','VALIDATION_STARTED.json','metrics.json']),'no real run artifacts')
r=dict(status='passed',count=len(checks),checks=checks,real_source_wav_header_reads=0,real_pcm_decodes=0,real_selection=0,network_bytes=0,model_calls=0)
(OUT/'synthetic_test_results.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r))
