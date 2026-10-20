from pathlib import Path
import csv,json,math,hashlib,wave,struct,time,signal,sys,os,datetime,traceback,importlib.abc
D=Path(__file__).resolve().parent;R=D.parents[1];OLD=R/'experiments/multitrajectory_oracle_heldout16';MAN=R/'experiments/rrsd_cv_repeat32/manifest.csv'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def wr(n,x):(D/n).write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')
def avg(x):return math.fsum(x)/len(x)
def main():
 assert not any((D/x).exists() for x in ['STARTED.json','metrics.json','failure.json']);t=time.monotonic();signal.signal(signal.SIGALRM,lambda *_:(_ for _ in ()).throw(TimeoutError('120s')));signal.alarm(120)
 with (D/'STARTED.json').open('x') as f:json.dump(dict(pid=os.getpid(),utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),driver_sha256=sha(__file__)),f)
 counts={'audio_decodes':0,'models':0,'ASR':0,'quality_recomputations':0,'network':0,'training':0,'audio_outputs':0,'resampling':0,'retries':0}
 try:
  class Deny(importlib.abc.MetaPathFinder):
   def find_spec(self,n,path=None,target=None):
    if n.split('.')[0] in {'numpy','torch','soundfile','whisper','torchaudio','pystoi','pesq','transformers'}:raise RuntimeError('forbidden import '+n)
  sys.meta_path.insert(0,Deny())
  def guard(e,a):
   if e.startswith('socket.') or e in {'subprocess.Popen','os.system','os.posix_spawn','os.fork'}:raise RuntimeError('forbidden '+e)
  sys.addaudithook(guard)
  cfg=json.loads((D/'config.yaml').read_text());assert sha(OLD/'metrics.json')==cfg['historical_metrics_sha256'];assert sha(MAN)==cfg['manifest_sha256'];old=json.loads((OLD/'metrics.json').read_text());manifest=list(csv.DictReader(MAN.open()))[16:32];assert len(manifest)==16
  sources=[];paths=set()
  for row in manifest:
   selected=[x for x in old['per_trajectory'] if x['filename']==row['filename']];assert len(selected)==4 and sorted(x['seed'] for x in selected)==[2026,2027,2028,2029];selected.sort(key=lambda x:x['seed']);assert all('sha256' not in x and 'wav_sha256' not in x for x in selected)
   for x in selected:assert Path(x['wav']).parent==Path(cfg['cache_directory']) and x['nfe']==30
   rs={'filename':row['filename'],'speaker':row['speaker'],'clean':row['clean'],'trajectories':selected};sources.append(rs);paths.add(row['clean']);paths.update(x['wav'] for x in selected)
  assert len(paths)==80
  protected=[OLD/'config.yaml',OLD/'metrics.json',OLD/'experiment_report.md',MAN,R/'experiments/multitrajectory_oracle_20261017/multitrajectory_oracle.py',R/'experiments/p8_vb_prospective_headroom_20261019/model_stage/metrics.json',R/'experiments/p8_vb_residual_direction_audit_20261019/metrics.json',D/'config.yaml',D/'analysis_plan.md']
  prior={str(p):sha(p) for p in protected};assert prior[str(protected[5])]=='577b9567aaad1a7ffc7f5d3fdae0b9c2dcf0da0bd89b86c0f5bbe14818d73f04';assert prior[str(protected[6])]=='06603c8fef1a7ee7b5a7bed89b159f6f97036e06e7d04c241524190892fe3e17'
  lock={}
  for p in sorted(paths):
   q=Path(p);assert q.is_file()
   with wave.open(p,'rb') as w:
    h=dict(rate=w.getframerate(),channels=w.getnchannels(),width=w.getsampwidth(),frames=w.getnframes(),compression=w.getcomptype());assert (h['rate'],h['channels'],h['width'],h['compression'])==(16000,1,2,'NONE')
   lock[p]=dict(bytes=q.stat().st_size,sha256=sha(q),header=h)
  wr('source_metadata_pre.json',prior);wr('source_PCM_locks.json',lock);wr('source_registry.json',{'source_rows':sources,'historical_byte_hash_gap':True,'claim':'Current80bytes only; no original generation WAV hashes exist in per_trajectory, cannot certify byte identity to generation time'})
  def decode(p):
   assert sha(p)==lock[p]['sha256']
   with wave.open(p,'rb') as w:b=w.readframes(w.getnframes())
   assert len(b)==2*lock[p]['header']['frames'];counts['audio_decodes']+=1;return [x[0]/32768 for x in struct.iter_unpack('<h',b)]
  rows=[]
  for src in sources:
   s=decode(src['clean']);bs=[decode(x['wav']) for x in src['trajectories']];assert all(len(x)==len(s) for x in bs);L=len(s);P=avg([x*x for x in s]);assert P>0;M=[math.fsum(v)/4 for v in zip(*bs)];ends=[math.fsum((b-c)**2 for b,c in zip(x,s))/L for x in bs];Eavg=avg(ends);Em=math.fsum((m-c)**2 for m,c in zip(M,s))/L;V=math.fsum(math.fsum((b-m)**2 for b,m in zip(x,M))/L for x in bs)/4;Vu=4*V/3;bias=Em-Vu/4
   vals={'E_average':Eavg,'E_mean':Em,'V_population4':V,'V_unbiased_under_IID':Vu,'finite4_bias2_est_raw_under_IID':bias};res=Eavg-Em-V;assert math.isclose(res,0,abs_tol=1e-13)
   rows.append({'filename':src['filename'],'speaker':src['speaker'],'samples':L,'P':P,'seed_endpoint_MSEs':dict(zip([str(x['seed']) for x in src['trajectories']],ends)),**vals,'normalized_by_P':{k:v/P for k,v in vals.items()},'identity_residual':res,'population_variance_fraction':V/Eavg if Eavg else None,'negative_bias2_est':bias<0})
   del s,bs,M
  strata={}
  for name,rs in [('all',rows)]+[('speaker_'+s,[r for r in rows if r['speaker']==s]) for s in sorted({r['speaker'] for r in rows})]:
   values={k:avg([r['normalized_by_P'][k] for r in rs]) for k in rows[0]['normalized_by_P']};ratios=[r['population_variance_fraction'] for r in rs if r['population_variance_fraction'] is not None];strata[name]={'files':len(rs),'mean_normalized':values,'mean_identity_residual':values['E_average']-values['E_mean']-values['V_population4'],'mean_file_variance_fraction':avg(ratios) if ratios else None,'pooled_normalized_variance_fraction':values['V_population4']/values['E_average'] if values['E_average'] else None,'negative_bias2_est_count':sum(r['negative_bias2_est'] for r in rs)}
  assert len(strata)==9
  for p,x in lock.items():
   assert Path(p).stat().st_size==x['bytes'] and sha(p)==x['sha256']
   with wave.open(p,'rb') as w:assert dict(rate=w.getframerate(),channels=w.getnchannels(),width=w.getsampwidth(),frames=w.getnframes(),compression=w.getcomptype())==x['header']
  assert prior=={p:sha(p) for p in prior};wr('source_metadata_post.json',prior)
  out={'status':'completed','seconds':time.monotonic()-t,'source_unique_WAVs':80,**counts,'files':rows,'strata':strata,'historical_original_byte_identity_proven':False,'scope':'finite cached fourseed Router/manualprepare_specs PCM, not currentVBnativeOFF/trueposterior; IID estimates conditional assumptions only'};wr('metrics.json',out);assert sum(p.stat().st_size for p in D.iterdir() if p.is_file())<2097152;print(json.dumps({'status':'completed','seconds':out['seconds'],'decodes':counts['audio_decodes'],'all':strata['all']}))
 except BaseException as e:wr('failure.json',{'status':'failed','error':repr(e),'traceback':traceback.format_exc(),'seconds':time.monotonic()-t,**counts});raise
 finally:signal.alarm(0)
if __name__=='__main__':main()
