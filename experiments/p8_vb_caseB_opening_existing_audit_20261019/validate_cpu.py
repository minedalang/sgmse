"""Independent source and full-FFT arithmetic validation, no audit import."""
import os,sys,pathlib,json,csv,hashlib,time,math
T=time.monotonic();D=pathlib.Path(__file__).resolve().parent
if (D/'independent_validation.json').exists():raise RuntimeError('No repeat validation')
def guard(e,a):
 if e.startswith('socket.') or e in ('subprocess.Popen','os.system','os.exec','os.posix_spawn'):raise RuntimeError('forbidden process/network')
 if e=='open' and isinstance(a[0],(str,bytes)) and isinstance(a[2],int) and a[2]&(os.O_WRONLY|os.O_RDWR|os.O_CREAT|os.O_TRUNC):
  if pathlib.Path(os.fsdecode(a[0])).resolve()!=D/'independent_validation.json':raise RuntimeError('validator write forbidden')
sys.addaudithook(guard)
import numpy as np
import soundfile as sf
assert not any(k.split('.')[0] in ('torch','whisper','transformers') for k in sys.modules)
def sha(p):return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
c=json.loads((D/'config.yaml').read_text());m=json.loads((D/'metrics.json').read_text());assert m['status']=='completed'
assert all(sha(p)==s for p,s in c['source_locks'].items());assert m['source_pre']==m['source_post']==c['source_locks']
P=pathlib.Path(c['parent']);L=P/'attempt02/listening';s=json.loads((L/'sources.json').read_text());arrays={}
for r in s['records']:
 if r['group']!='B':continue
 arrays[r['arm']+'_pcm']=sf.read(str(L/r['copy']),dtype='float64')[0]
 e=r['evidence'];arrays[r['arm']+'_raw']=(np.load(e['raw_npy'],allow_pickle=False).reshape(-1) if 'raw_npy' in e else np.fromfile(e['raw_float32'],dtype='<f4')).astype('float64')
 assert sha(L/r['copy'])==sha(r['source'])==r['sha256']
for typ in ('pcm','raw'):arrays['off_minus_frontend_'+typ]=arrays['off_'+typ]-arrays['frontend_'+typ]
n=sf.read(m['noise_recipe']['path'],dtype='float64')[0][80000:147653]*m['noise_recipe']['gain']*m['noise_recipe']['scale']
count=0
for name in ('fixed_interval.csv','all_frames20ms.csv'):
 rows=list(csv.DictReader((D/name).open()))
 assert len(rows)==(70 if name=='fixed_interval.csv' else 1696)
 for row in rows:
  a=int(row['start_sample']);b=int(row['end_sample_exclusive']);x=arrays[row['arm']][a:b];N=len(x);assert N==int(row['samples'])
  assert int(row['zero_count'])==sum(x==0);assert math.isclose(float(row['rms'])**2,float(np.dot(x,x)/N),rel_tol=2e-13,abs_tol=1e-25)
  assert float(row['peak'])==max(abs(x));assert float(row['minimum'])==min(x) and float(row['maximum'])==max(x);assert float(row['zero_fraction'])==sum(x==0)/N
  # Full complex FFT, including both signed frequencies, independently checks real-FFT weighting.
  z=np.fft.fft(x);freq=np.abs(np.fft.fftfreq(N,1/16000));power=abs(z)**2/N**2
  for key,mask in [('power_0_4k',freq<4000),('power_4_7p5k',(freq>=4000)&(freq<7500)),('power_7p5_8k',freq>=7500)]:
   assert math.isclose(float(row[key]),float(power[mask].sum()),rel_tol=3e-11,abs_tol=1e-22),(name,row['arm'],a,key)
  if name=='fixed_interval.csv' and row.get('alpha'):
   nn=math.fsum(float(v)*float(v) for v in n[a:b]);yy=math.fsum(float(v)*float(v) for v in x);yn=math.fsum(float(u)*float(v) for u,v in zip(x,n[a:b]));assert math.isclose(float(row['alpha']),yn/nn,rel_tol=1e-11,abs_tol=1e-15);assert math.isclose(float(row['signed_cosine']),yn/math.sqrt(nn*yy),rel_tol=1e-11,abs_tol=1e-15);assert math.isclose(float(row['projected_power_fraction']),yn*yn/(nn*yy),rel_tol=1e-11,abs_tol=1e-20)
  count+=1
 if name=='all_frames20ms.csv':
  for arm in [k for k in arrays if not k.startswith('off_minus')]:
   rr=[r for r in rows if r['arm']==arm];assert len(rr)==212 and [int(r['start_sample']) for r in rr]==list(range(0,67653,320));assert int(rr[-1]['samples'])==133 and int(rr[-1]['end_sample_exclusive'])==67653
result=dict(status='passed',validated_rows=count,fixed_intervals=7,frames_per_arm=212,last_frame_samples=133,source_locks=len(c['source_locks']),metrics_sha256=sha(D/'metrics.json'),method='Independent full complex FFT bands; dot-product RMS; Python math.fsum noise projection; full source hashes',seconds=time.monotonic()-T,model_loads=0,ASR_calls=0,network_bytes=0,audio_outputs=0)
(D/'independent_validation.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
