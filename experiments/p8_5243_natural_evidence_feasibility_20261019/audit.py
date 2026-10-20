"""CPU source/mixture/static replay feasibility audit. No torch/model imports."""
from pathlib import Path
import io,json,hashlib,time,datetime,subprocess
import numpy as np
import soundfile as sf
H=Path(__file__).resolve().parent;R=H.parent.parent;T=time.monotonic()
def sha(b):return hashlib.sha256(b).hexdigest()
def read(p):return json.loads(p.read_text())
def dump(n,d):(H/n).write_text(json.dumps(d,indent=2,allow_nan=False)+'\n')
def stats(x):return {'rms':float(np.sqrt(np.mean(x*x))),'mean_square':float(np.mean(x*x)),'peak':float(np.max(np.abs(x))),'nonzero_samples':int(np.count_nonzero(x)),'samples':len(x)}
def audio(p):
 b=Path(p).read_bytes();f=sf.SoundFile(io.BytesIO(b));meta={'path':str(Path(p).resolve()),'sha256':sha(b),'bytes':len(b),'sample_rate':f.samplerate,'channels':f.channels,'frames':len(f),'format':f.format,'subtype':f.subtype};x=f.read(dtype='float64');f.close();assert meta['sample_rate']==16000 and meta['channels']==1;return x,meta
assert not (H/'metrics.json').exists(),'Existing completed audit; refuse overwrite'
cfg={'experiment_id':H.name,'status':'frozen_cpu_audit','fileid':5243,'clip_seconds':[18.75,20.75],'target_neighborhood_seconds':[19.35,19.55],'target_interval_is_human_boundary':False,'budget_seconds':180,'artifact_cap_bytes':5000000,'inference':False,'training':False,'downloads':False,'output_audio':False};dump('config.yaml',cfg)
git_before=subprocess.check_output(['git','status','--porcelain'],cwd=R).decode();(H/'git_status_before.txt').write_text(git_before)
align=R/'experiments/p8_e1_candidate_listening_aligned_20261020';record=next(r for r in read(align/'metrics.json')['records'] if r['fileid']==5243)
arrays={};sources={};clips={}
for arm,path in record['source_paths'].items():
 x,m=audio(path);arrays[arm]=x;m['matches_registered_source_hash']=m['sha256']==record['source_sha256'][arm];assert m['matches_registered_source_hash'];assert len(x)==480000;sources[arm]=m
 matches=list((align/'clips').glob('*fileid_5243*context_'+arm+'.wav'));assert len(matches)==1
 clip,cm=audio(matches[0]);cm['source_sample_range']=[300000,332000];cm['exact_pcm_crop_match']=np.array_equal(clip,x[300000:332000]);cm['maximum_crop_difference']=float(np.max(np.abs(clip-x[300000:332000])));assert cm['exact_pcm_crop_match'];clips[arm]=cm
noise_path=Path(record['source_paths']['clean']).parent.parent/'noise/noise_fileid_5243.wav';noise_exists=noise_path.exists();mix={};noise_meta={'path':str(noise_path),'exists':noise_exists}
if noise_exists:
 n,nm=audio(noise_path);noise_meta.update(nm);assert n.shape==arrays['clean'].shape
 c=arrays['clean'];y=arrays['noisy'];lsb=1/32768;delta=y-c-n
 # Three independent PCM16 quantization errors each <1 LSB: conservative 3-LSB bound.
 assert all(sources[k]['subtype']=='PCM_16' for k in ['clean','noisy']) and nm['subtype']=='PCM_16'
 bound=3*lsb
 def segment(a,b):
  cc,nn,yy=c[a:b],n[a:b],y[a:b];err=yy-cc-nn;coef=np.linalg.lstsq(np.stack([cc,nn],axis=1),yy,rcond=None)[0];res=yy-coef[0]*cc-coef[1]*nn
  return {'samples':[a,b],'seconds':[a/16000,b/16000],'clean':stats(cc),'noise':stats(nn),'noisy':stats(yy),'mixture_error':stats(err),'mixture_error_max_lsb':float(np.max(np.abs(err))/lsb),'mixture_error_rms_lsb':float(np.sqrt(np.mean(err*err))/lsb),'samples_exceeding_3lsb':int(np.count_nonzero(np.abs(err)>bound)),'pcm_quantization_bound':bound,'passes_3lsb_decomposition':bool(np.max(np.abs(err))<=bound),'least_squares_clean_gain':float(coef[0]),'least_squares_noise_gain':float(coef[1]),'least_squares_residual':stats(res),'paired_component_snr_db':float(10*np.log10(np.mean(cc*cc)/np.mean(nn*nn))),'ls_fitted_component_snr_db':float(10*np.log10(np.mean((coef[0]*cc)**2)/np.mean((coef[1]*nn)**2)))}
 mix={name:segment(a,b) for name,a,b in [('global',0,480000),('context',300000,332000),('target_neighborhood',309600,312800)]}
 mix['interpretation']='SNR uses supplied paired component powers, not consonant-isolated SNR. Least-squares is descriptive; it does not create or replace noise. Three-LSB bound assumes independent PCM16 quantization with each error <1LSB, and is conservative even for truncation.'
 clean_nonzero=np.flatnonzero(c);mix['clean_source_support']={'first_nonzero_sample':int(clean_nonzero[0]),'last_nonzero_sample_inclusive':int(clean_nonzero[-1]),'total_nonzero_samples':len(clean_nonzero),'exact_zero_samples':int(np.count_nonzero(c==0)),'clean_energy':float(c@c),'target_neighborhood_clean_energy_fraction':float(c[309600:312800]@c[309600:312800]/(c@c)),'meaning':'Exact nonzero PCM support is not a phonetic boundary or VAD truth.'}
# Static provenance; hash source text but inherit checkpoint lock without loading weights.
refpaths=[align/'metrics.json',align/'config.yaml',R/'experiments/p8_frontend_backend_audit_20261020/config.yaml',R/'experiments/p8_frontend_backend_audit_20261020/metrics.json',R/'experiments/p8_frontend_backend_audit_20261020/experiment_report.md',R/'experiments/p8_frontend_backend_audit_20261020/user_listening_observation.md',R/'experiments/p8_q_a_ctc_fixed_time_v2_20261018/experiment_report.md',R/'joint_enhance_by_dir.py',R/'sgmse/joint_model.py',R/'experiments/p8_10720_pairing_forensics_20261018/replay_off/attempt02/run_replay.py',R/'experiments/p8_10720_pairing_forensics_20261018/replay_off/attempt02/metrics.json',R/'experiments/p8_10720_tail_causal_pilot_20261018/run_pilot.py',R/'experiments/p8_10720_tail_causal_pilot_20261018/config.yaml',R/'experiments/p8_y_pre_mask_counterfactual_20261020/run_probe.py']
refpaths+=list((R/'experiments/p8_frontend_backend_audit_20261020/logs').glob('*.log'))
# Preserve relevant historical metadata/report references, not output audio.
for directory in (R/'experiments').glob('p8*'):
 if directory==H:continue
 if any(word in directory.name for word in ['tail_','y_pre_mask','noisy_condition','5243','ctc_fixed_time_v2','e1_candidate_listening_aligned','frontend_backend_audit']):
  for name in ['config.yaml','experiment_report.md']:
   if (directory/name).is_file():refpaths.append(directory/name)
refs=[];matches=[]
for p in sorted(set(refpaths)):
 if not p.exists():continue
 b=p.read_bytes();text=b.decode(errors='replace');refs.append({'path':str(p),'bytes':len(b),'sha256':sha(b)})
 for i,line in enumerate(text.splitlines(),1):
  if any(w in line.lower() for w in ['5243','seed','manual_seed','blake2','19.45','19.35','10.0','10.5','rrsd','mask','frontend_raw','file_seed','pcm','no_ema','checkpoint','y_pre =','_frontend_forward']):matches.append({'path':str(p),'line':i,'text':line})
dump('source_references.json',refs);dump('matched_references.json',matches)
basename=Path(record['source_paths']['off']).name;base=2026;seed=(base+int.from_bytes(hashlib.blake2b(basename.encode(),digest_size=4).digest(),'little'))%2**32
original_basename=Path(record['source_paths']['noisy']).name;wrong_seed=(base+int.from_bytes(hashlib.blake2b(original_basename.encode(),digest_size=4).digest(),'little'))%2**32
inputp=R/'experiments/p8_frontend_backend_audit_20261020/inputs/noisy'/basename
checkpoint=R/'joint_train_logs/sb61i0gj_RRSD/epoch=118-step=80000.ckpt'
locked=read(R/'experiments/p8_vcv_speech_component_ablation_20261019/metrics.json')['sha256_locks'][str(checkpoint)]
raw_candidates=[]
for directory in [R/'experiments/p8_frontend_backend_audit_20261020',R/'experiments/p8_10720_pairing_forensics_20261018']:
 raw_candidates += [str(p) for p in directory.rglob('*.npy') if '5243' in str(p) or 'frontend' in p.name]
assert all(sha(Path(r['path']).read_bytes())==r['sha256'] for r in refs)
assert all(sha(Path(m['path']).read_bytes())==m['sha256'] for m in list(sources.values())+list(clips.values())+[noise_meta] if m.get('exists',True))
git_after=subprocess.check_output(['git','status','--porcelain'],cwd=R).decode();(H/'git_status_after.txt').write_text(git_after)
d={'status':'completed_cpu_feasibility_audit','source_record':record,'sources_rehashed':sources,'clip_checks':clips,'noise_source':noise_meta,'mixture':mix,'original_runtime':{'base_seed':base,'source_basename':basename,'input_symlink_path':str(inputp),'input_symlink_exists':inputp.exists(),'input_symlink_resolves_to':str(inputp.resolve()),'effective_file_seed':seed,'unprefixed_original_basename':original_basename,'wrong_unprefixed_seed':wrong_seed,'use_ema':True,'sampler':'pc','predictor':'reverse_diffusion','N':30,'corrector':'none','snr':0.5,'rrsd':False,'source':'original config/off_retry.log plus current driver lines277-343; seeding precedes frontend'},'checkpoint':{'path':str(checkpoint),'sha256':locked,'hash_status':'inherited existing immutable lock from completed five-case metrics, not rehashed in this CPU audit','exists':checkpoint.exists(),'bytes':checkpoint.stat().st_size},'cached_raw_5243_frontend_candidates':raw_candidates,'exact_replay_achieved':False,'inference_calls':0,'training_calls':0,'source_locks_reverified':True,'git_status_unchanged':git_before==git_after,'wall_seconds':time.monotonic()-T,'artifact_bytes':0}
assert d['wall_seconds']<180;dump('metrics.json',d)
print(json.dumps({'seed':seed,'wrong_unprefixed_seed':wrong_seed,'mixture':mix,'raw_frontend_candidates':raw_candidates,'wall_seconds':d['wall_seconds']},indent=2))
