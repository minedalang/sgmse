"""Read-only audio/HTTP verification. Writes new audit artifacts only; refuses overwrite."""
from pathlib import Path
import csv,json,hashlib,re,urllib.request,urllib.parse,io,subprocess,datetime
import numpy as np
import soundfile as sf
from scipy.signal import correlate
ROOT=Path('/data/coding/sgmse'); E=ROOT/'experiments'; OUT=Path(__file__).resolve().parent
if (OUT/'metrics.json').exists(): raise RuntimeError('Existing results: refuse overwrite')
sha=lambda b:hashlib.sha256(b).hexdigest()
def read(p):
 b=Path(p).read_bytes(); a,sr=sf.read(io.BytesIO(b),dtype='float64'); inf=sf.info(io.BytesIO(b));return a,dict(path=str(p),sha256=sha(b),sample_rate=sr,channels=inf.channels,subtype=inf.subtype,format=inf.format,samples=len(a),duration_s=len(a)/sr,rms=float(np.sqrt(np.mean(a*a))),peak=float(np.max(abs(a))))
def search(q,s):
 q=q-q.mean(); m=len(q); cs=np.r_[0,np.cumsum(s)]; cs2=np.r_[0,np.cumsum(s*s)]; den=np.maximum(cs2[m:]-cs2[:-m]-(cs[m:]-cs[:-m])**2/m,0)*np.dot(q,q)
 r=correlate(s,q,mode='valid',method='fft')/np.sqrt(np.maximum(den,1e-60)); k=int(np.argmax(abs(r)));return dict(best_start_sample=k,best_start_s=k/16000,correlation=float(r[k]),correlation_at_expected_13p75=float(r[220000]) if len(r)>220000 else None)
def lag(a,b,rate=16000):
 # xcorr convention a[t+lag] vs b[t], minimum overlap half duration
 n=min(len(a),len(b));a=a[:n];b=b[:n];lags=np.arange(-n+1,n);mask=abs(lags)<=n//2; ls=lags[mask]; cc=correlate(a,b,mode='full',method='fft')[mask]
 ac=np.r_[0,np.cumsum(a)];bc=np.r_[0,np.cumsum(b)];aa=np.r_[0,np.cumsum(a*a)];bb=np.r_[0,np.cumsum(b*b)]; a0=np.maximum(ls,0);b0=np.maximum(-ls,0); count=n-abs(ls);a1=a0+count;b1=b0+count
 av=ac[a1]-ac[a0];bv=bc[b1]-bc[b0];r=(cc-av*bv/count)/np.sqrt(np.maximum((aa[a1]-aa[a0]-av*av/count)*(bb[b1]-bb[b0]-bv*bv/count),1e-60));k=int(np.argmax(abs(r)));return dict(best_lag_samples=int(ls[k]),best_lag_s=float(ls[k]/rate),correlation=float(r[k]),zero_lag_correlation=float(r[np.flatnonzero(ls==0)[0]]),minimum_overlap_s=float(n/2/rate))
def envelope(a):return np.sqrt(np.mean(a[:len(a)//160*160].reshape(-1,160)**2,axis=1))
D=dict(experiment_id=OUT.name,status='running',started_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),git_status=subprocess.getoutput('git -C /data/coding/sgmse status --short'),failures=[],http=[],sources={},clips={},clip_full_source_search={},full_pair_lags={},cross_file_matrix=[])
(OUT/'config.yaml').write_text('experiment_id: '+OUT.name+'\nfileid: 10720\nmode: read_only_pairing_forensics\ncontext_seconds: [13.75, 15.75]\nsample_range: [220000, 252000]\nclip_search: full_30s_valid_centered_normalized_correlation\nfull_lag_search: minimum_15s_overlap_centered_normalized_correlation\nenvelope: 10ms_RMS\nnetwork: localhost_8766_only\n')
record=next(r for r in json.loads((E/'p8_e1_candidate_listening_aligned_20261020/metrics.json').read_text())['records'] if r['fileid']==10720)
S={}; C={};pref='03_fileid_10720_14.50-15.00s_context_'
for arm,p in record['source_paths'].items():
 S[arm],D['sources'][arm]=read(p);D['sources'][arm]['matches_recorded_sha256']=D['sources'][arm]['sha256']==record['source_sha256'][arm]
 cp=E/'p8_e1_candidate_listening_aligned_20261020/clips'/(pref+arm+'.wav');C[arm],info=read(cp);ex=S[arm][220000:252000];info.update(exact_float_pcm_equality=bool(np.array_equal(C[arm],ex)),max_absolute_difference=float(np.max(abs(C[arm]-ex))),least_squares_scale=float(np.dot(C[arm],ex)/np.dot(ex,ex)),expected_sample_range=[220000,252000]);D['clips'][arm]=info
for page in ['p8_10720_content_identity_audit_20261018/listening.html','p8_e1_candidate_listening_aligned_20261020/listening.html','p8_e1_candidate_listening_20261018/listening.html']:
 url='http://127.0.0.1:8766/'+page
 try:
  with urllib.request.urlopen(url,timeout=10) as r: b=r.read();entry=dict(url=url,status=r.status,sha256=sha(b),equals_disk=b==(E/page).read_bytes())
  html=b.decode(); paths=re.findall(r'src="([^"]+10720[^"]+\.wav)"',html)
  if 'content_identity' in page:
   directory=re.search(r"const directory='([^']+)'",html)[1];prefix=re.search(r"const prefix='([^']+)'",html)[1];paths=[directory+prefix+a+'.wav' for a in S]
   entry['player_windows_s']=re.findall(r'data-start="([^"]+)" data-end="([^"]+)"',html);entry['default_gain']=1;entry['rms_equalization_default']=False
  entry['audio_routes']=[]
  for rel in paths:
   au=urllib.parse.urljoin(url,rel);disk=(E/Path(page).parent/rel).resolve()
   with urllib.request.urlopen(au,timeout=10) as r: ab=r.read();entry['audio_routes'].append(dict(url=au,status=r.status,path=str(disk),sha256=sha(ab),equals_disk=ab==disk.read_bytes()))
  D['http'].append(entry)
 except Exception as exc:D['failures'].append(dict(stage='http',url=url,error=repr(exc)))
for ca,q in C.items():D['clip_full_source_search'][ca]={sa:search(q,s) for sa,s in S.items()}
for aa,a in S.items():
 for ba,b in S.items():
  if aa==ba:continue
  D['full_pair_lags'][aa+'_vs_'+ba]=dict(waveform=lag(a,b),rms_envelope=lag(envelope(a),envelope(b),100))
rows=list(csv.DictReader((E/'p8_frontend_backend_audit_20261020/manifest.csv').open()))
for row in rows:
 inputs={a:row[a+'_path'] for a in ['clean','noisy']}
 basename=f"{int(row['index']):02d}_"+row['noisy_basename']
 inputs['y_pre']=str(E/'p8_frontend_backend_audit_20261020/outputs/y_pre'/basename)
 for ia,p in inputs.items():
  b,info=read(p)
  for oa in ['off','router']:
   D['cross_file_matrix'].append(dict(output_arm=oa,input_fileid=int(row['fileid']),input_arm=ia,input_info=info,waveform=lag(S[oa],b),rms_envelope=lag(envelope(S[oa]),envelope(b),100)))
D['status']='completed' if not D['failures'] else 'completed_with_failures';D['completed_at_utc']=datetime.datetime.now(datetime.timezone.utc).isoformat()
(OUT/'metrics.json').write_text(json.dumps(D,indent=2)+'\n')
print(json.dumps({'status':D['status'],'clips':D['clips'],'full_pair_lags':D['full_pair_lags']},indent=2))
