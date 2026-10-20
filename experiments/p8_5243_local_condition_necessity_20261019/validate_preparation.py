"""CPU-only independent frozen clean/time selection and condition checks."""
from pathlib import Path
import json,hashlib
import numpy as np
import soundfile as sf
H=Path(__file__).resolve().parent
def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
s=read(H/'selection_frozen.json');m=read(H/'preparation_metrics.json');cfg=read(H/'config.yaml');assert sha(H/'selection_frozen.json')==m['selection_sha256']==cfg['frozen_selection_sha256'];assert sha(H/'config.yaml')==m['config_sha256'];assert sha(H/'prepare_selection.py')==m['driver_sha256']
old=read(H/'preparation_failure_record.json')['selection_and_config_already_frozen'];assert all(sha(H/n)==h for n,h in old.items())
assert all(sha(p)==h for p,h in s['source_locks'].items())
cleanpath=next(p for p in s['source_locks'] if p.endswith('/clean_fileid_5243.wav'));noisypath=next(p for p in s['source_locks'] if '/noisy/' in p and p.endswith('5243.wav'));clean,sr=sf.read(cleanpath,dtype='float64');noisy,nsr=sf.read(noisypath,dtype='float32');assert sr==nsr==16000
w=np.ones(3200);r=.5-.5*np.cos(np.linspace(0,np.pi,320));w[:320]=r;w[-320:]=r[::-1];assert hashlib.sha256(w.tobytes()).hexdigest()==s['taper_float64_sha256'];et=float(np.sum((w*clean[309600:312800])**2));assert et==s['target']['weighted_clean_energy']
assert [r['start_sample'] for r in s['all_candidate_windows']]==list(range(300000,328801,320))
for row in s['all_candidate_windows']:
 a=row['start_sample'];b=row['end_sample_exclusive'];energy=float(np.sum((w*clean[a:b])**2));assert energy==row['weighted_clean_energy'];side='left' if b<=306400 else 'right' if a>=316000 else None;assert row['side']==side and row['eligible']==(side is not None and energy>=et)
for side in ['left','right']:
 expected=min([r for r in s['all_candidate_windows'] if r['side']==side and r['eligible']],key=lambda r:(r['distance_to_target_interval_samples'],r['start_sample']));assert s['selected_controls'][side]==expected
for name,r in {'target':s['target'],**s['selected_controls']}.items():
 a=r['start_sample'];b=r['end_sample_exclusive'];alpha=r.get('alpha',r.get('alpha_if_energy_eligible'));x=noisy.copy();x[a:b]=(noisy[a:b].astype(np.float64)-alpha*w*clean[a:b]).astype(np.float32);assert hashlib.sha256(x.tobytes()).hexdigest()==m['dry_checks'][name]['condition_float32_sha256'];assert np.array_equal(x[:a],noisy[:a]) and np.array_equal(x[b:],noisy[b:]);assert np.max(np.abs(x))==np.max(np.abs(noisy));assert alpha<=1
assert cfg['future_calls']=={'frontend':0,'backend_OFF_N30':4,'existing_recognizer_short_calls':4};assert cfg['future_budget_seconds']==240 and cfg['future_artifact_cap_bytes']==30000000 and not cfg['execution_authorized']
assert not (H/'metrics.json').exists() and not (H/'experiment_report.md').exists();assert not list(H.glob('*.wav')) and not list(H.glob('*.npy'))
result={'status':'passed','frozen_grid_windows_checked':len(s['all_candidate_windows']),'clean_only_control_selection_verified':True,'in_memory_condition_hashes_reconstructed':3,'frozen_selection_config_unchanged_since_initial_write':True,'source_locks_rehashed':len(s['source_locks']),'checkpoint_loaded':False,'model_calls':0,'reserved_run_outputs_absent':True};(H/'preparation_validation.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
