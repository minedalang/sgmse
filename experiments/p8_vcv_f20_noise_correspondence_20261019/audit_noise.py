"""Fixed saved-waveform zero-alignment noise audit; NumPy CPU, no model calls.
Reproduce in a fresh directory: copy this script there and execute with Python.
Repository root is inferred from the experiments directory layout.
"""
from pathlib import Path
import hashlib,json,time,subprocess
import numpy as np

H=Path(__file__).resolve().parent; R=H.parent.parent; T=time.monotonic()
assert not (H/'metrics.json').exists(), 'Refuse completed experiment overwrite'
A=R/'experiments/p8_vcv_speech_component_ablation_20261019'
F=R/'experiments/p8_q_vcv_speaker_replication_20261019'
P=R/'experiments/p8_vcv_f20_tail_existing_audit_20261019'
CID='fresh__N2_snr-6__f20aka1'; SR=16000
def read(p): return json.loads(p.read_text())
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def ah(a): return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()
def dump(name,data): (H/name).write_text(json.dumps(data,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
prior=read(P/'metrics.json'); recipe=read(F/'outputs/recipe_frozen.json')
case=next(x for x in read(A/'selection_frozen.json')['cases'] if x['case_id']==CID)
records=[x for x in read(A/'metrics.json')['cases'] if x['case_id']==CID]
native=next(x for x in records if not x['ablated']); ablate=next(x for x in records if x['ablated'])
proof=next(x for x in read(A/'outputs/source_proofs.json') if x['case_id']==CID)
rec=next(x for x in recipe['recipes'] if x['sample_id']=='f20aka1' and x['condition']=='N2_snr-6')
row=next(x for x in recipe['rows'] if x['sample_id']=='f20aka1')
old=R/'experiments/p8_q_vcv_fixed_noise_20261018'
noise_rec=next(x for x in read(old/'inputs/recipe_frozen.json')['noise'] if x['noise_id']=='N2')
noise_path=old/noise_rec['array_path']
paths={'clean':F/'outputs/f20aka1_clean.npy','y_pre':F/case['frontend_audio']['raw_path'],
       'native_off':A/native['raw_path'],'ablated_off':A/ablate['raw_path']}
sources=list(dict.fromkeys([Path(p) for p in prior['source_locks_before_after']]+list(paths.values())+
 [noise_path,old/'inputs/recipe_frozen.json',P/'metrics.json',P/'config.yaml',P/'audit_tail.py',
  P/'experiment_report.md',A/'listening/manual_listener_spectral_followup_20261019.json',
  A/'listening/manual_listener_feedback_completion_20261019.json',Path(__file__)]))
locks={str(p):sha(p) for p in sources}
assert all(locks[p]==v for p,v in prior['source_locks_before_after'].items())
git_before=subprocess.check_output(['git','status','--porcelain'],cwd=R).decode()
(H/'git_status_before.txt').write_text(git_before)
intervals={'after_declared_support':[14881,32000],'after_declared_support_plus100ms':[16481,32000]}
assert intervals==prior['intervals']
config={'case_id':CID,'sample_rate':SR,'intervals':intervals,'CPU_only':True,'model_calls':0,
 'training_calls':0,'downloads':0,'audio_outputs':0,'budget_seconds':120,'artifact_cap_bytes':1000000,
 'reference':'Frozen float64 physical N2 noise = recorded noise_gain * frozen unit waveform; output physical scale',
 'bands':{'fullband':'all RFFT bins, 0 <= f <= 8000','lowband':'0 <= f < 4000','highband':'4000 <= f <= 7500'},
 'band_boundary':'No low/high overlap. Frequencies >7500 remain only in fullband.',
 'DFT':'Unwindowed finite-interval RFFT; weights 1 for DC/Nyquist (if present), 2 otherwise; /N^2 mean products',
 'primary':['signed_rho','least_squares_gain','projection_fraction_rho_squared','residual_mean_square'],
 'secondary':'cosine of weighted per-bin spectral powers, phase insensitive',
 'forbidden':'No optimized lag, filter, kernel, threshold, window, or classifier; no waveform output'}
dump('config.yaml',config)
arrays={name:np.load(p,allow_pickle=False) for name,p in paths.items()}
unit=np.load(noise_path,allow_pickle=False)['waveform']; noise=rec['noise_gain']*unit
condition_noise=noise.astype(np.float32); noisy=(arrays['clean']+noise).astype(np.float32)
assert ah(noise)==proof['physical_noise_float64_sha256']
assert ah(noise/proof['refscale'])==proof['normalized_physical_noise_sha256']
assert ah(condition_noise)==proof['noise_only_float32_sha256']
assert ah(noisy)==proof['noisy_float32_sha256']==rec['model_input_float32_sha256']
assert ah(arrays['clean'])==proof['clean_float64_sha256']
start=row['source_onset_sample']; end=start+row['resampled_samples']; assert [start,end]==[8000,14881]
outside=np.ones(32000,bool); outside[start:end]=False
assert np.count_nonzero(arrays['clean'][outside])==0
assert np.array_equal(noisy[outside],condition_noise[outside])
for k in ['anchor_raw_sha256','normalized_anchor_sha256','refscale','seed','cpu_rng_sha256','cuda_rng_sha256','initial_xT_sha256','normalized_physical_noise_sha256']:
 assert native[k]==ablate[k]
assert native['nfe']==ablate['nfe']==30
arrays['ablated_minus_native']=arrays['ablated_off'].astype(np.float64)-arrays['native_off'].astype(np.float64)
assert all(x.shape==(32000,) and np.isfinite(x).all() for x in arrays.values())

def compare(x,n,band):
 x=np.asarray(x,np.float64); n=np.asarray(n,np.float64); N=len(x)
 X=np.fft.rfft(x); Z=np.fft.rfft(n); freq=np.fft.rfftfreq(N,1/SR)
 w=np.full(len(X),2.); w[0]=1
 if N%2==0: w[-1]=1
 masks={'fullband':np.ones(len(X),bool),'lowband':freq<4000,'highband':(freq>=4000)&(freq<=7500)}
 mask=masks[band]; weights=w[mask]/N**2; X=X[mask]; Z=Z[mask]
 xx=float(np.sum(weights*np.abs(X)**2)); nn=float(np.sum(weights*np.abs(Z)**2))
 cross=float(np.sum(weights*np.real(X*np.conj(Z)))); assert nn>0
 gain=cross/nn; residual=X-gain*Z; rr=float(np.sum(weights*np.abs(residual)**2))
 orth=float(np.sum(weights*np.real(residual*np.conj(Z)))); projected=gain*gain*nn
 rho=cross/np.sqrt(xx*nn) if xx>0 else None; fraction=rho*rho if rho is not None else None
 assert abs(orth)<=1e-12*np.sqrt(max(xx,1e-300)*nn)+1e-30
 assert abs(xx-projected-rr)<=1e-12*max(xx,1e-300)+1e-30
 if fraction is not None: assert 0<=fraction<=1+1e-12
 px=weights*np.abs(X)**2; pn=weights*np.abs(Z)**2
 cosine=float(np.dot(px,pn)/(np.linalg.norm(px)*np.linalg.norm(pn))) if xx>0 else None
 if band=='fullband':
  assert abs(xx-float(np.mean(x*x)))<=1e-12*max(xx,1e-300)+1e-30
  assert abs(cross-float(np.mean(x*n)))<=1e-12*np.sqrt(max(xx,1e-300)*nn)+1e-30
 return {'samples':N,'bins':int(mask.sum()),'first_bin_hz':float(freq[mask][0]),'last_bin_hz':float(freq[mask][-1]),
 'output_mean_square':xx,'noise_mean_square':nn,'cross_mean_product':cross,'signed_rho':rho,
 'least_squares_gain':gain,'projection_fraction_rho_squared':fraction,'projected_mean_square':projected,
 'residual_mean_square':rr,'residual_noise_mean_product':orth,'decomposition_absolute_error':abs(xx-projected-rr),
 'phase_insensitive_spectral_power_cosine':cosine,'undefined_reason':None if xx>0 else 'Output has exactly zero energy; rho, rho squared and spectral cosine undefined; LS gain and residual are zero.'}

metrics={name:{arm:{band:compare(x[a:b],noise[a:b],band) for band in config['bands']} for arm,x in arrays.items()} for name,(a,b) in intervals.items()}
# Fixed identity/opposite/zero checks exercise sign, power invariance, decomposition and undefined values.
t=np.arange(1024)/SR; test=np.sin(2*np.pi*5000*t)
assert abs(compare(test,test,'fullband')['signed_rho']-1)<1e-12
assert abs(compare(-test,test,'highband')['signed_rho']+1)<1e-12
assert abs(compare(-test,test,'highband')['phase_insensitive_spectral_power_cosine']-1)<1e-12
assert compare(np.zeros_like(test),test,'fullband')['signed_rho'] is None
assert all(sha(p)==v for p,v in locks.items())
git_after=subprocess.check_output(['git','status','--porcelain'],cwd=R).decode()
(H/'git_status_after.txt').write_text(git_after)
out={'status':'completed','case_id':CID,'metrics':metrics,'source_locks_before_after':locks,
 'all_sources_unchanged':True,'previous_audit_locks_match':True,'physical_noise_gain':rec['noise_gain'],
 'reference_array_sha256':ah(noise),'actual_condition_noise_float32_sha256':ah(condition_noise),
 'normalized_physical_noise_sha256':ah(noise/proof['refscale']),
 'condition_proof':{'all_padding_native_ablated_conditions_bitwise_equal':True,'padding_samples':int(outside.sum()),
 'tail_intervals_clean_exact_zero':True,'actual_float32_noise_quantization_max_abs':float(np.max(np.abs(condition_noise.astype(np.float64)-noise))),
 'interval_condition_hashes':{key:ah(condition_noise[a:b]) for key,(a,b) in intervals.items()}},
 'model_calls':0,'training_calls':0,'downloads':0,'audio_outputs':0,'numerical_checks_passed':True,
 'git_status_unchanged':git_before==git_after,'CPU_wall_seconds':time.monotonic()-T}
assert out['CPU_wall_seconds']<120
dump('metrics.json',out)
lines=['# Fixed case5 noise correspondence audit','',
 'CPU-only saved-waveform audit. The physical float64 N2 noise, recorded gain, normalized-noise hash, clean/noisy/noise-only condition hashes, and prior audit locks all match. Native and ablated local conditions are bitwise identical in both fixed tails; clean is exactly zero. No inference, training, downloads or audio outputs.',
 '', 'Projection uses signed zero-alignment inner products, no mean subtraction or lag fitting. Each band is selected from the same unwindowed finite-interval DFT. Lowband is [0,4000) Hz, highband [4000,7500] Hz; fullband also contains >7500 Hz. Weighted Parseval products are mean products in physical output scale. Rho squared is the fraction explained by one unconstrained signed scalar times the original noise. LS residual orthogonality, energy decomposition, bounds, fullband time-domain agreement and identity/opposite/zero checks pass.',
 '', '|Interval start sample|Arm|Band|Signed rho|LS gain|Projected power %|Residual mean square|Power-spectrum cosine|',
 '|---:|---|---|---:|---:|---:|---:|---:|']
for name,(a,b) in intervals.items():
 for arm,bands in metrics[name].items():
  for band,m in bands.items():
   values=[f'{m[k]:.9g}' if m[k] is not None else 'undefined' for k in ['signed_rho','least_squares_gain']]
   fraction='undefined' if m['projection_fraction_rho_squared'] is None else f"{100*m['projection_fraction_rho_squared']:.9g}"
   cos='undefined' if m['phase_insensitive_spectral_power_cosine'] is None else f"{m['phase_insensitive_spectral_power_cosine']:.9g}"
   lines.append(f"|{a}|{arm}|{band}|{values[0]}|{values[1]}|{fraction}|{m['residual_mean_square']:.9g}|{cos}|")
lines += ['', 'Clean has exactly zero energy: correlation, projected fraction and power-spectrum cosine are undefined, while LS gain and residual power are zero. Spectral cosine compares powers only and cannot establish phase-aligned retention.',
 '', 'A high signed correlation supports only a zero-alignment, linearly coherent component. Low correlation leaves nonlinear filtering, time-varying gains and phase changes unresolved; it cannot prove a new phoneme or semantic hallucination. A scalar projection does not identify the entire generation mechanism. This is one previously selected, clean-aware OOD whole-VCV condition ablation with the speech-containing frontend anchor fixed. The same listener spectral followup is provenance, not an independent label. No new classifier or listener measurement is introduced.',
 '', f"Reproduce by copying audit_noise.py into a fresh experiments subdirectory and running /data/miniconda/envs/torch/bin/python audit_noise.py. Completed audit CPU wall time: {out['CPU_wall_seconds']:.6f} s. Source files unchanged. Git status unchanged across numerical run: {out['git_status_unchanged']} (concurrent work may affect repository-level status). Exact full precision results and hashes are in metrics.json."]
(H/'experiment_report.md').write_text('\n'.join(lines)+'\n')
size=sum(p.stat().st_size for p in H.rglob('*') if p.is_file())
out['artifact_bytes']=size; dump('metrics.json',out)
size=sum(p.stat().st_size for p in H.rglob('*') if p.is_file()); out['artifact_bytes']=size; dump('metrics.json',out)
assert size<1000000
print(json.dumps({'CPU_wall_seconds':out['CPU_wall_seconds'],'artifact_bytes':size,'primary':{
 name:{arm:metrics[name][arm]['highband'] for arm in ['y_pre','native_off','ablated_off','ablated_minus_native']} for name in intervals}},indent=2))
