"""Single bounded standard-library audit; writes no audio."""
from pathlib import Path
import math,json,hashlib,wave,struct,time,signal,sys,os,datetime,traceback,importlib.abc
D=Path(__file__).resolve().parent;R=D.parents[1];SRC=R/'experiments/p8_vb_prospective_headroom_20261019'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(n,v):(D/n).write_text(json.dumps(v,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
def load(p):return json.loads(Path(p).read_text())
def mean(v):return math.fsum(v)/len(v)
def groups(rows):
 out={'all':rows}
 for n in ['N1','N2']:out['noise_'+n]=[r for r in rows if r['noise']==n]
 for q in [-18,0]:out['snr_'+str(q)]=[r for r in rows if r['snr_db']==q]
 for n in ['N1','N2']:
  for q in [-18,0]:out[n+'_'+str(q)]=[r for r in rows if r['noise']==n and r['snr_db']==q]
 for key in ['speaker','id']:
  for v in sorted({r[key] for r in rows}):out[('speaker_' if key=='speaker' else 'utterance_')+v]=[r for r in rows if r[key]==v]
 return out
def main():
 assert not any((D/n).exists() for n in ['STARTED.json','metrics.json','failure.json'])
 t=time.monotonic();signal.signal(signal.SIGALRM,lambda *_:(_ for _ in ()).throw(TimeoutError('60s audit')));signal.alarm(60)
 with (D/'STARTED.json').open('x') as f:json.dump({'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'pid':os.getpid(),'script_sha256':sha(__file__)},f)
 state={'status':'running','source_pcm_decodes':0,'audio_outputs':0,'models':0,'ASR':0,'quality_calls':0,'network':0,'training':0,'resampling':0,'retries':0}
 try:
  class Block(importlib.abc.MetaPathFinder):
   def find_spec(self,fullname,path=None,target=None):
    if fullname.split('.')[0] in {'numpy','torch','soundfile','whisper','torchaudio','pystoi','pesq','transformers'}:raise RuntimeError('forbidden scientific import '+fullname)
  sys.meta_path.insert(0,Block())
  def audit(e,a):
   if e.startswith('socket.') or e in {'subprocess.Popen','os.system','os.posix_spawn','os.fork'}:raise RuntimeError('network/subprocess forbidden '+e)
  sys.addaudithook(audit)
  reg=load(SRC/'input_stage/input_registry.json');m=load(SRC/'model_stage/metrics.json');cfg=load(D/'config.yaml')
  assert sha(SRC/'input_stage/input_registry.json')==cfg['input_registry_sha256'];assert sha(SRC/'model_stage/metrics.json')==cfg['source_metrics_sha256']
  # Preserve all existing source experiment bytes, including metadata/reports/input and raw arrays;
  # hashing is not decoding; only selected72 WAVs are ever decoded.
  preserve={str(p):sha(p) for p in SRC.rglob('*') if p.is_file() and not p.is_symlink()}
  preserve[str(D/'analysis_plan.md')]=sha(D/'analysis_plan.md');preserve[str(D/'config.yaml')]=sha(D/'config.yaml');write('source_preservation_pre.json',preserve)
  native={r['basename']:r for r in m['native_cases']};controls={g['id']:g for g in reg['groups']};paths={}
  for c in reg['cases']:
   paths[str(SRC/'input_stage'/controls[c['id']]['scaledclean']['wav'])]=controls[c['id']]['scaledclean']['wav_sha256']
   for arm in ['frontend','off']:paths[str(SRC/'model_stage'/native[c['basename']][arm+'_pcm']['path'])]=native[c['basename']][arm+'_pcm']['sha256']
  assert len(paths)==72
  cache={};locks={}
  for name,h in paths.items():
   p=Path(name);assert sha(p)==h
   with wave.open(name,'rb') as w:
    header={'channels':w.getnchannels(),'width':w.getsampwidth(),'rate':w.getframerate(),'frames':w.getnframes(),'compression':w.getcomptype()};assert header['channels']==1 and header['width']==2 and header['rate']==16000 and header['compression']=='NONE';b=w.readframes(header['frames']);assert len(b)==2*header['frames']
   cache[name]=[v[0]/32768 for v in struct.iter_unpack('<h',b)];state['source_pcm_decodes']+=1;locks[name]={'sha256':h,'bytes':p.stat().st_size,'header':header}
  rows=[]
  for c in reg['cases']:
   n=native[c['basename']];sp=str(SRC/'input_stage'/controls[c['id']]['scaledclean']['wav']);fp=str(SRC/'model_stage'/n['frontend_pcm']['path']);bp=str(SRC/'model_stage'/n['off_pcm']['path']);s=cache[sp];f=cache[fp];b=cache[bp];assert len(s)==len(f)==len(b)==c['samples']
   e=[x-y for x,y in zip(f,s)];d=[y-x for x,y in zip(f,b)];P=mean([v*v for v in s]);assert P>0;E0=mean([v*v for v in e]);A=mean([x*y for x,y in zip(e,d)]);en=mean([v*v for v in d]);E1=mean([(x-y)**2 for x,y in zip(b,s)]);u=-A/en if en>0 else None;a=max(0.,min(1.,u)) if u is not None else 0.;eo=E0+2*a*A+a*a*en;PF=mean([v*v for v in f]);beta=mean([x*y for x,y in zip(d,f)])/PF if PF>0 else None;gain=[beta*x for x in f] if beta is not None else [0.]*len(f);rest=[x-y for x,y in zip(d,gain)];G=mean([v*v for v in gain]);Q=mean([v*v for v in rest]);cross=mean([x*y for x,y in zip(gain,rest)]);cg=2*mean([x*y for x,y in zip(e,gain)])+G;cr=2*mean([x*y for x,y in zip(e,rest)])+Q
   assert eo<=min(E0,E1)+1e-14
   orig=next(q for q in m['quality']['per_case'] if q['basename']==c['basename'])['off_minus_frontend']['si_sdr']
   vals={'P':P,'E0':E0,'E1':E1,'A':A,'D':en,'E1_minus_E0':E1-E0,'E_oracle':eo,'oracle_minus_frontend':eo-E0,'oracle_minus_OFF':eo-E1}
   rows.append({'basename':c['basename'],'id':c['id'],'speaker':c['speaker'],'noise':c['noise'],'snr_db':c['snr_db'],'samples':len(s),'paths':{'S':sp,'F':fp,'B':bp},**vals,'normalized_by_P':{k:v/P for k,v in vals.items() if k!='P'},'identity_residual':E1-(E0+2*A+en),'alpha_unconstrained':u,'alpha_oracle':a,'alpha_class':'no_change' if en==0 else 'zero' if a==0 else 'one' if a==1 else 'interior','PF':PF,'gain_beta':beta,'gain_energy':G,'rest_energy':Q,'gain_rest_inner':cross,'gain_MSE_contribution':cg,'rest_MSE_contribution':cr,'part_contribution_residual':(E1-E0)-(cg+cr),'rest_energy_fraction':Q/en if en>0 else None,'saved_original_OFF_minus_frontend_si_sdr_db':orig})
  strata={}
  for label,rs in groups(rows).items():
   a_sum=math.fsum(x['A']/x['P'] for x in rs);d_sum=math.fsum(x['D']/x['P'] for x in rs);u=-a_sum/d_sum if d_sum>0 else None;a=max(0.,min(1.,u)) if u is not None else 0.
   strata[label]={'cases':len(rs),'common_alpha_unconstrained':u,'common_alpha_oracle':a,'mean_frontend_normalized_MSE':mean([x['E0']/x['P'] for x in rs]),'mean_OFF_normalized_MSE':mean([x['E1']/x['P'] for x in rs]),'mean_common_alpha_normalized_MSE':mean([(x['E0']+2*a*x['A']+a*a*x['D'])/x['P'] for x in rs]),'mean_case_oracle_normalized_MSE':mean([x['E_oracle']/x['P'] for x in rs]),'alpha_classes':{v:sum(x['alpha_class']==v for x in rs) for v in ['no_change','zero','interior','one']},'mean_rest_energy_fraction':mean([x['rest_energy_fraction'] for x in rs if x['rest_energy_fraction'] is not None]) if any(x['rest_energy_fraction'] is not None for x in rs) else None,'mean_gain_MSE_contribution_normalized':mean([x['gain_MSE_contribution']/x['P'] for x in rs]),'mean_rest_MSE_contribution_normalized':mean([x['rest_MSE_contribution']/x['P'] for x in rs]),'mean_saved_OFF_minus_frontend_si_sdr_db':mean([x['saved_original_OFF_minus_frontend_si_sdr_db'] for x in rs if x['saved_original_OFF_minus_frontend_si_sdr_db'] is not None])}
  post={p:sha(p) for p in preserve};assert preserve==post
  # Header and byte locks re-read after computation; only headers, not second PCM decode.
  for p,item in locks.items():
   assert Path(p).stat().st_size==item['bytes'] and sha(p)==item['sha256']
   with wave.open(p,'rb') as w:assert {'channels':w.getnchannels(),'width':w.getsampwidth(),'rate':w.getframerate(),'frames':w.getnframes(),'compression':w.getcomptype()}==item['header']
  write('source_pcm_locks.json',locks);write('source_preservation_post.json',post);state.update(status='completed',cases=rows,strata=strata,source_unique_WAVs=len(paths),preserved_files=len(preserve),seconds=time.monotonic()-t,source_preservation_equal=True)
  write('metrics.json',state);assert sum(p.stat().st_size for p in D.rglob('*') if p.is_file())<2097152
  print(json.dumps({'status':'completed','seconds':state['seconds'],'decodes':state['source_pcm_decodes'],'cases':len(rows),'all':strata['all']}))
 except BaseException as ex:
  state.update(status='failed',seconds=time.monotonic()-t,error=repr(ex),traceback=traceback.format_exc());write('failure.json',state);raise
 finally:signal.alarm(0)
if __name__=='__main__':main()
