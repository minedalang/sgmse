"""Pre-register matched clean-energy position controls; NumPy/SoundFile CPU only."""
from pathlib import Path
import json,hashlib,time,datetime,subprocess
import numpy as np
import soundfile as sf
H=Path(__file__).resolve().parent;R=H.parent.parent;B=R/'experiments/p8_5243_native_replay_gate_20261019/attempt02';F=R/'experiments/p8_5243_natural_evidence_feasibility_20261019';Q=R/'experiments/p8_q_a_ctc_fixed_time_v2_20261018';T=time.monotonic()
assert not (H/'preparation_metrics.json').exists(),'Refuse completed preparation'
assert not (H/'selection_frozen.json').exists() or (H/'preparation_failure_record.json').exists(),'Refuse unrecorded partial selection'
def read(p):return json.loads(p.read_text())
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
def ah(x):return hashlib.sha256(np.ascontiguousarray(x).tobytes()).hexdigest()
def dump(n,d):
 text=json.dumps(d,indent=2,ensure_ascii=False,allow_nan=False,default=lambda x:x.item())+'\n'
 if n in ('selection_frozen.json','config.yaml') and (H/n).exists():
  assert (H/n).read_text()==text,'Frozen preparation changed: '+n
 else:(H/n).write_text(text)
base=read(B/'metrics.json');audit=read(F/'metrics.json');assert base['status']=='passed';assert (base['frontend_calls'],base['backend_calls'],base['nfe'],base['trace_calls'])==(1,1,30,30)
assert base['pre_hashes']==base['post_hashes']==base['config']['source_locks']
gitbefore=subprocess.check_output(['git','status','--short'],cwd=R,text=True);(H/'git_status_before.txt').write_text(gitbefore)
paths={k:Path(audit['source_record']['source_paths'][k]) for k in ['clean','noisy']};paths['noise']=Path(audit['noise_source']['path'])
sourcepaths=list(paths.values())+[B/'metrics.json',B/'artifact_validation.json',B/'config.yaml',B/'run_replay.py',B/'outputs/frontend_raw.npy',B/'outputs/backend_raw.npy',F/'metrics.json',F/'experiment_report.md',Q/'metrics.json',Q/'config.yaml',Q/'fixed_time_probe.py',Q/'frame_probabilities.npz']
sourcepaths+=list((B/'outputs').glob('rng*.npy'))
# Rehash all current source/core locks except large checkpoint, whose actual successful runtime pre/post hash is inherited.
checkpoint=base['config']['checkpoint']
sourcepaths+=[Path(p) for p in base['config']['source_locks'] if p!=checkpoint]
sourcepaths=list(dict.fromkeys(sourcepaths));locks={str(p):sha(p) for p in sourcepaths}
for p,h in base['config']['source_locks'].items():
 if p!=checkpoint:assert locks[p]==h
for arm in ['clean','noisy']:assert locks[str(paths[arm])]==audit['source_record']['source_sha256'][arm]
assert locks[str(paths['noise'])]==audit['noise_source']['sha256']
raw=np.load(B/'outputs/frontend_raw.npy',allow_pickle=False);assert ah(raw)==base['frontend_raw']['sha256']==base['backend_anchor']['sha256']
back=np.load(B/'outputs/backend_raw.npy',allow_pickle=False);assert ah(back)==base['backend_raw']['sha256']
clean,sr=sf.read(paths['clean'],dtype='float64');noisy,nsr=sf.read(paths['noisy'],dtype='float32');noise,nnsr=sf.read(paths['noise'],dtype='float64');assert sr==nsr==nnsr==16000 and clean.shape==noisy.shape==noise.shape==(480000,)
assert all(sf.info(p).subtype=='PCM_16' for p in paths.values())
SR=16000;N=3200;FADE=320;target_start=309600;target_end=312800;context_start=300000;context_end=332000
w=np.ones(N,dtype=np.float64);ramp=.5-.5*np.cos(np.linspace(0,np.pi,FADE));w[:FADE]=ramp;w[-FADE:]=ramp[::-1];assert w[0]==w[-1]==0 and np.array_equal(w,w[::-1])
def E(a):return float(np.sum((w*clean[a:a+N])**2,dtype=np.float64))
et=E(target_start);assert et>0
candidates=[]
for a in range(context_start,context_end-N+1,320):
 b=a+N;side='left' if b<=306400 else 'right' if a>=316000 else None;energy=E(a);eligible=side is not None and energy>=et
 candidates.append({'start_sample':a,'end_sample_exclusive':b,'start_s':a/SR,'end_s':b/SR,'side':side,'weighted_clean_energy':energy,'energy_eligible':energy>=et,'position_eligible':side is not None,'eligible':eligible,'distance_to_target_interval_samples':target_start-b if side=='left' else a-target_end if side=='right' else None,'alpha_if_energy_eligible':float(np.sqrt(et/energy)) if energy>=et else None})
selected={}
for side in ['left','right']:
 eligible=[r for r in candidates if r['side']==side and r['eligible']]
 if eligible:selected[side]=dict(min(eligible,key=lambda r:(r['distance_to_target_interval_samples'],r['start_sample'])))
selection={'status':'feasible' if len(selected)==2 else 'unfeasible','selection_inputs':'clean PCM16 energy and fixed times only; no output/CTC preference','sample_rate':SR,'target':{'start_sample':target_start,'end_sample_exclusive':target_end,'start_s':19.35,'end_s':19.55,'alpha':1.,'weighted_clean_energy':et},'context_samples':[context_start,context_end],'candidate_start_grid_samples':320,'candidate_grid_anchor_sample':context_start,'window_samples':N,'internal_fade_samples':FADE,'fade_formula':'0.5-0.5*cos(linspace(0,pi,320)), endpoints inclusive; symmetric internal final320; plateau1','taper_float64_sha256':ah(w),'position_rule':'LEFT end<=19.15s;RIGHT start>=19.75s; both wholly within18.75–20.75s','energy_rule':'Econtrol=sum((w*clean)^2)>=Etarget;alpha=sqrt(Etarget/Econtrol)<=1','deterministic_selection':'closest interval distance to target per side, then earliest start; stop if either side unavailable','all_candidate_windows':candidates,'selected_controls':selected,'source_locks':locks,'checkpoint_lock_inherited':{'path':checkpoint,'sha256':base['pre_hashes'][checkpoint],'provenance':'successful attempt02 actual pre/post hashes; not reread/loaded by this preparation'},'frozen_original_anchor':{'path':str(B/'outputs/frontend_raw.npy'),'array_sha256':ah(raw),'dtype':str(raw.dtype),'shape':list(raw.shape)},'original_seed':369124093,'original_reference_peak':float(np.max(np.abs(noisy)))}
checks={};quant_bound=audit['mixture']['global']['pcm_quantization_bound'];original_residual=noisy.astype(np.float64)-clean-noise;assert np.max(np.abs(original_residual))<=quant_bound
if selection['status']=='feasible':
 variants={'target':selection['target'],**selected}
 for name,row in variants.items():
  a=row['start_sample'];b=row['end_sample_exclusive'];alpha=row['alpha'] if name=='target' else row['alpha_if_energy_eligible'];edit=alpha*w*clean[a:b];condition=noisy.copy();condition[a:b]=(noisy[a:b].astype(np.float64)-edit).astype(np.float32)
  outside=np.ones(480000,bool);outside[a:b]=False;assert np.array_equal(condition[outside],noisy[outside]);assert np.isfinite(condition).all();assert float(np.max(np.abs(condition)))==float(np.max(np.abs(noisy)))
  remain_clean=clean[a:b]-edit;residual=condition[a:b].astype(np.float64)-remain_clean-noise[a:b];noise_change=residual-original_residual[a:b];round_bound=np.finfo(np.float32).eps*max(float(np.max(np.abs(noisy))),float(np.max(np.abs(edit))))
  assert np.max(np.abs(noise_change))<=round_bound and np.max(np.abs(residual))<=quant_bound+round_bound
  actual_edit=noisy[a:b].astype(np.float64)-condition[a:b].astype(np.float64);actual_energy=float(actual_edit@actual_edit);ideal_energy=float(edit@edit)
  checks[name]={'window_samples':[a,b],'window_seconds':[a/SR,b/SR],'alpha':alpha,'ideal_edit_energy':ideal_energy,'target_energy':et,'ideal_energy_absolute_error':abs(ideal_energy-et),'float32_actual_edit_energy':actual_energy,'float32_energy_absolute_error':abs(actual_energy-et),'float32_energy_relative_error':abs(actual_energy-et)/et,'all_finite':True,'outside_window_bitwise_identical':True,'global_peak_bitwise_value_unchanged':True,'condition_float32_sha256':ah(condition),'maximum_removed_clean_contribution':float(np.max(np.abs(edit))),'noise_residual_change_max':float(np.max(np.abs(noise_change))),'noise_residual_float32_roundoff_bound':round_bound,'physical_noise_hash':ah(noise),'mixture_residual_max':float(np.max(np.abs(residual))),'mixture_quantization_bound':quant_bound,'mixture_bound_plus_roundoff':quant_bound+round_bound}
 assert max(c['float32_energy_relative_error'] for c in checks.values())<1e-5
config={'experiment_id':H.name,'status':'prepared_selection_only_no_inference','execution_authorized':False,'scientific_design':'local direct-condition known-clean necessity with matched-energy position controls','target_seconds':[19.35,19.55],'source_full_samples':480000,'source_sample_rate':SR,'frozen_selection_file':'selection_frozen.json','frozen_selection_sha256':None,'condition_formula':'copy originalfloat32 noisy; within selectedwindow assignfloat32(noisy_float64-alpha*w*known_clean_PCM16); untouched outside','clean_gain_fitting':False,'anchor':'frozen attempt02 frontend_raw.npy unchanged','rng':'frozen original effective seed369124093 and before-backend/before-sampler states; exact native gate required before edits','future_calls':{'frontend':0,'backend_OFF_N30':4,'existing_recognizer_short_calls':4},'future_runs':['native_exact_gate','target','left','right'],'future_budget_seconds':240,'future_artifact_cap_bytes':30000000,'future_gate':'native full raw arrays, historical serialized fullOFF and clip must reproduce before target/controls; trace/RNG/reference guards matched','future_interpretation':'single natural prompted /d/ event; target200ms bracket is not isolated manually segmented /d/; controls perturb equal L2 energy at different content positions, not phonetic equivalence','no_training':True,'no_downloads':True}
dump('selection_frozen.json',selection);config['frozen_selection_sha256']=sha(H/'selection_frozen.json');dump('config.yaml',config)
assert all(sha(p)==h for p,h in locks.items());after=subprocess.check_output(['git','status','--short'],cwd=R,text=True);(H/'git_status_after.txt').write_text(after)
pm={'status':'prepared_only' if selection['status']=='feasible' else 'selection_unfeasible_stop','selection_status':selection['status'],'candidate_count':len(candidates),'eligible_left':sum(c['eligible'] and c['side']=='left' for c in candidates),'eligible_right':sum(c['eligible'] and c['side']=='right' for c in candidates),'selected_controls':selected,'target_weighted_clean_energy':et,'dry_checks':checks,'source_locks_valid_before_after':True,'source_lock_count':len(locks),'checkpoint_loaded':False,'checkpoint_hash_inherited':True,'model_imports':0,'frontend_calls':0,'backend_calls':0,'recognizer_calls':0,'audio_variants_written':False,'git_status_unchanged':gitbefore==after,'wall_seconds':time.monotonic()-T,'driver_sha256':sha(__file__),'selection_sha256':sha(H/'selection_frozen.json'),'config_sha256':sha(H/'config.yaml')};dump('preparation_metrics.json',pm)
print(json.dumps({'status':pm['status'],'target_energy':et,'controls':selected,'dry_checks':checks,'wall_seconds':pm['wall_seconds']},indent=2,default=lambda x:x.item()))
