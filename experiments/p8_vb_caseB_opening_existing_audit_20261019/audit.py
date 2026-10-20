"""Reproduce frozen existing-audio measurements; CPU only, no audio outputs."""
import os,sys,time,json,hashlib,pathlib,io,csv,signal,sqlite3
T=time.monotonic(); D=pathlib.Path(__file__).resolve().parent
if (D/'STARTED.json').exists(): raise RuntimeError('No repeat execution authorized')
(D/'STARTED.json').write_text(json.dumps({'unix':time.time(),'command':sys.argv}))
signal.signal(signal.SIGALRM,lambda *_: (_ for _ in ()).throw(TimeoutError('60s cap')));signal.alarm(60)
def guard(event,args):
 if event.startswith('socket.') or event in ('subprocess.Popen','os.system','os.exec','os.posix_spawn'): raise RuntimeError('network/process forbidden: '+event)
 if event=='open' and isinstance(args[0],(str,bytes)) and isinstance(args[2],int) and args[2] & (os.O_WRONLY|os.O_RDWR|os.O_CREAT|os.O_TRUNC):
  p=pathlib.Path(os.fsdecode(args[0])).resolve()
  if D not in p.parents or p.suffix.lower() in ('.wav','.flac','.mp3','.npy','.f32le'): raise RuntimeError('write outside audit or audio forbidden')
sys.addaudithook(guard)
class NoModels:
 def find_spec(self,fullname,*args):
  if fullname.split('.')[0] in ('torch','whisper','transformers','torchaudio'): raise RuntimeError('model import forbidden')
sys.meta_path.insert(0,NoModels())
import numpy as np
import soundfile as sf

def sha(p): return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
def readj(p): return json.loads(pathlib.Path(p).read_text())
def save(name,x):
 (D/name).write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')
 assert sum(p.stat().st_size for p in D.rglob('*') if p.is_file())<=2000000

def wav(p):
 x,fs=sf.read(str(p),dtype='float64');assert fs==16000 and x.ndim==1;return x

def band(x):
 n=len(x); z=np.fft.rfft(x);w=np.full(len(z),2.);w[0]=1
 if n%2==0:w[-1]=1
 power=(z.real*z.real+z.imag*z.imag)*w/(n*n);f=np.fft.rfftfreq(n,1/16000)
 v=[float(power[f<4000].sum()),float(power[(f>=4000)&(f<7500)].sum()),float(power[f>=7500].sum())]
 assert abs(sum(v)-float(np.mean(x*x)))<=1e-14+1e-11*float(np.mean(x*x))
 return v

def stats(x):
 b=band(x);return dict(samples=len(x),rms=float(np.sqrt(np.mean(x*x))),peak=float(np.max(np.abs(x))),minimum=float(x.min()),maximum=float(x.max()),zero_count=int(np.count_nonzero(x==0)),zero_fraction=float(np.mean(x==0)),power_0_4k=b[0],power_4_7p5k=b[1],power_7p5_8k=b[2])
def projection(y,n):
 yy=float(y@y);nn=float(n@n);yn=float(y@n)
 return dict(alpha=yn/nn if nn else None,signed_cosine=yn/(yy*nn)**.5 if yy and nn else None,projected_power_fraction=yn*yn/(yy*nn) if yy and nn else None,undefined_reason='zero_denominator' if not yy or not nn else None)

m={'status':'started','failures':[],'model_loads':0,'ASR_calls':0,'training_calls':0,'network_bytes':0,'audio_outputs':0}
try:
 c=readj(D/'config.yaml');P=pathlib.Path(c['parent']);L=P/'attempt02/listening'; reg=readj(P/'input_registry.json');old=readj(P/'attempt02/metrics.json');src=readj(L/'sources.json')
 locks=c['source_locks']; pre={p:sha(p) for p in locks};assert pre==locks; m['source_pre']=pre
 assert sha(L/'user_feedback_20261019.json')==c['feedback_sha256']
 db=sqlite3.connect('file:/data/codex-state/state_5.sqlite?mode=ro',uri=True)
 provider=db.execute('select id,model_provider from threads where agent_path=? order by created_at desc limit 1',('/root/experiment_worker_vb_opening',)).fetchone();db.close();assert provider[1]=='codex_tools_compat';m['provider']={'thread':provider[0],'provider':provider[1]}
 case=next(r for r in reg['cases'] if r['basename']==c['case']);group=next(g for g in reg['groups'] if g['id']==case['id']);noise=next(r for r in reg['noise_sources'] if r['id']=='N2');native=next(r for r in old['native_cases'] if r['basename']==c['case'])
 assert native['nfe']==30 and native['native_length_match']==dict(input=67653,reference=67653,output=67653,operation='identity',left=0,right=0)
 assert native['backend_actual_norm_factor']==native['noisy_peak_native_reference']==0.899993896484375
 assert group['noise_start_sample']==80000 and group['noise_end_sample_exclusive']==147653
 n=wav(noise['path'])[80000:147653]*case['gain']*case['scale']; original=wav(next(r for r in src['records'] if r['group']=='B' and r['arm']=='clean')['evidence']['source_wave']); clean64=original*case['scale']; mixture64=(original+wav(noise['path'])[80000:147653]*case['gain'])*case['scale']
 arrays={};proof={}
 for r in src['records']:
  if r['group']!='B':continue
  arm=r['arm'];play=L/r['copy'];assert play.read_bytes()==pathlib.Path(r['source']).read_bytes() and sha(play)==r['sha256'];x=wav(play);assert len(x)==67653;arrays[arm+'_pcm']=x
  evidence=r['evidence'];raw=np.load(evidence['raw_npy'],allow_pickle=False).reshape(-1) if 'raw_npy' in evidence else np.fromfile(evidence['raw_float32'],dtype='<f4'); assert len(raw)==67653 and raw.dtype==np.float32
  arrays[arm+'_raw']=raw.astype('float64')
  if arm in ('frontend','off'):
   b=io.BytesIO();sf.write(b,raw,16000,format='WAV',subtype='PCM_16');assert b.getvalue()==play.read_bytes();assert hashlib.sha256(raw.tobytes()).hexdigest()==evidence['raw_array_sha256'];proof[arm]={'memory_libsndfile_bytes_exact':True,'raw_shape':list(np.load(evidence['raw_npy'],allow_pickle=False).shape)}
  else:
   target=clean64 if arm=='clean' else mixture64
   assert np.array_equal(raw,target.astype('<f4'));assert np.array_equal(x,np.floor(target*32768)/32768)
   proof[arm]={'float32_exact':True,'pcm_floor_exact':True,'max_pcm_vs_raw':float(np.max(np.abs(x-raw))),'max_pcm_vs_reconstruction':float(np.max(np.abs(x-target))),'max_float32_rounding':float(np.max(np.abs(raw-target)))}
  proof[arm]['header']=dict(rate=16000,channels=sf.info(str(play)).channels,subtype=sf.info(str(play)).subtype,frames=len(x))
 assert np.max(np.abs(arrays['noisy_raw']-arrays['clean_raw']-n))<4e-8
 m['source_proof']=proof;m['native_geometry']={k:native[k] for k in ('seed','nfe','native_length_match','backend_actual_norm_factor','noisy_peak_native_reference')};m['noise_recipe']=dict(path=noise['path'],start=80000,end=147653,gain=case['gain'],scale=case['scale'],raw_mixture_residual_max=float(np.max(np.abs(arrays['noisy_raw']-arrays['clean_raw']-n))),pcm_mixture_residual_max=float(np.max(np.abs(arrays['noisy_pcm']-clean64-n))))
 calibrations={}
 for name,f,expected in [('DC',0,[1.,0.,0.]),('sine1k',1000,[.5,0.,0.]),('sine6k',6000,[0.,.5,0.]),('sine7p8k',7800,[0.,0.,.5]),('Nyquist',8000,[0.,0.,1.])]:
  t=np.arange(1600);x=np.ones(1600) if f==0 else np.cos(2*np.pi*f*t/16000);v=band(x);assert np.allclose(v,expected,atol=1e-12,rtol=0);calibrations[name]=v
 assert projection(np.zeros(10),np.ones(10))['signed_cosine'] is None and projection(np.ones(10),np.zeros(10))['alpha'] is None
 m['calibration']=calibrations
 fixed=[]
 for label,a,b in c['intervals']:
  for arm,x in arrays.items():
   row=dict(interval=label,start_sample=a,end_sample_exclusive=b,start_seconds=a/16000,end_seconds=b/16000,arm=arm,**stats(x[a:b]))
   if arm.split('_')[0] in ('frontend','off'):row.update(projection(x[a:b],n[a:b]))
   fixed.append(row)
  for typ in ('raw','pcm'):
   fixed.append(dict(interval=label,start_sample=a,end_sample_exclusive=b,start_seconds=a/16000,end_seconds=b/16000,arm='off_minus_frontend_'+typ,**stats((arrays['off_'+typ]-arrays['frontend_'+typ])[a:b]),**projection((arrays['off_'+typ]-arrays['frontend_'+typ])[a:b],n[a:b])))
 def writecsv(name,rows):
  fields=list(dict.fromkeys(k for r in rows for k in r));f=(D/name).open('w');w=csv.DictWriter(f,fields);w.writeheader();w.writerows(rows);f.close()
 writecsv('fixed_interval.csv',fixed)
 frames=[]
 for i,a in enumerate(range(0,67653,320)):
  b=min(a+320,67653)
  for arm,x in arrays.items():frames.append(dict(frame=i,start_sample=a,end_sample_exclusive=b,arm=arm,**stats(x[a:b])))
 writecsv('all_frames20ms.csv',frames)
 m['fixed_intervals']=fixed;m['frames_per_arm']=212;m['last_frame_samples']=133;m['versions']=dict(numpy=np.__version__,soundfile=sf.__version__,libsndfile=sf.__libsndfile_version__)
 post={p:sha(p) for p in locks};assert pre==post;m['source_post']=post;m['status']='completed'
except BaseException as e:
 import traceback
 m['status']='failed';m['failures'].append(dict(type=type(e).__name__,message=str(e),traceback=traceback.format_exc()))
finally:
 m['execution_seconds']=time.monotonic()-T;save('metrics.json',m)
if m['status']!='completed':raise SystemExit(1)
print(json.dumps({'status':m['status'],'seconds':m['execution_seconds'],'source_locks':len(locks),'frames':212}))
