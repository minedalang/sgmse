"""Independent inverse-RFFT verification after audit_noise.py; no models/audio outputs."""
from pathlib import Path
import json,hashlib,time
import numpy as np
H=Path(__file__).resolve().parent; R=H.parent.parent; T=time.monotonic()
def read(p): return json.loads(p.read_text())
def dump(p,x): p.write_text(json.dumps(x,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
M=read(H/'metrics.json'); assert not (H/'independent_verification.json').exists()
assert all(hashlib.sha256(Path(p).read_bytes()).hexdigest()==h for p,h in M['source_locks_before_after'].items())
P=read(R/'experiments/p8_vcv_f20_tail_existing_audit_20261019/metrics.json')
src={r['arm']:Path(r['path']) for r in P['sources']}
F=R/'experiments/p8_q_vcv_speaker_replication_20261019'
rec=next(r for r in read(F/'outputs/recipe_frozen.json')['recipes'] if r['sample_id']=='f20aka1' and r['condition']=='N2_snr-6')
old=R/'experiments/p8_q_vcv_fixed_noise_20261018'
nr=next(r for r in read(old/'inputs/recipe_frozen.json')['noise'] if r['noise_id']=='N2')
noise=rec['noise_gain']*np.load(old/nr['array_path'])['waveform']
arrays={k:np.load(p).astype(np.float64) for k,p in src.items()}
arrays['ablated_minus_native']=arrays['ablated_off']-arrays['native_off']; count=0
for name,(a,b) in P['intervals'].items():
 n=noise[a:b]; N=len(n); f=np.fft.rfftfreq(N,1/16000)
 for arm,x in arrays.items():
  for band in ['fullband','lowband','highband']:
   mask=np.ones(len(f),bool) if band=='fullband' else f<4000 if band=='lowband' else (f>=4000)&(f<=7500)
   xb=np.fft.irfft(np.fft.rfft(x[a:b])*mask,n=N); nb=np.fft.irfft(np.fft.rfft(n)*mask,n=N)
   rho=np.dot(xb,nb)/np.sqrt(np.dot(xb,xb)*np.dot(nb,nb)) if np.any(xb) else None
   gain=np.dot(xb,nb)/np.dot(nb,nb); residual=xb-gain*nb; m=M['metrics'][name][arm][band]
   assert (rho is None and m['signed_rho'] is None) or np.isclose(rho,m['signed_rho'],rtol=1e-12,atol=1e-15)
   assert np.isclose(gain,m['least_squares_gain'],rtol=1e-12,atol=1e-15)
   assert np.isclose(np.mean(residual*residual),m['residual_mean_square'],rtol=1e-12,atol=1e-25)
   count+=1
V={'status':'passed','method':'Independent band inverse-RFFT then time-domain dot products',
 'arm_interval_band_checks':count,'checked':['signed_rho','least_squares_gain','residual_mean_square'],
 'source_locks_rechecked':True,'CPU_wall_seconds':time.monotonic()-T,
 'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
dump(H/'independent_verification.json',V)
text='''

The highband increase has little fixed-zero-lag scalar correspondence to the original noise: ablated rho is 0.01042551 / 0.01031635 in the two intervals, with only 0.0108691% / 0.0106427% of its highband power projected onto the original noise. Native values are 0.00939370 / 0.00659056 (0.00882416% / 0.00434355%). For ablated minus native, the fractions are 0.00900371% / 0.00985294%. Ablated highband power-spectrum cosine is 0.463399 / 0.468958, versus native 0.206833 / 0.204777; these phase-insensitive power similarities do not establish waveform retention. Lowband native/ablated rho remains small (0.02440894 / 0.02408076 first interval; 0.01609174 / 0.01968503 guarded). Thus the larger measured highband power is chiefly outside this single aligned-noise projection; it remains unresolved whether nonlinear or phase-changing noise processing explains the sound. No semantic insertion claim follows.

Independent verification: verify_noise.py reconstructs each selected band with inverse-RFFT and recomputes all 30 arm × interval × band cases in the time domain. Signed rho, LS gain and residual power agree, including explicit zero-energy clean handling; all original source locks pass again. Run audit_noise.py then verify_noise.py in a fresh destination. Both scripts refuse overwriting their completed results.
'''
with (H/'experiment_report.md').open('a') as f: f.write(text)
M['independent_verification']=V
for _ in range(4):
 M['artifact_bytes']=sum(p.stat().st_size for p in H.rglob('*') if p.is_file())
 dump(H/'metrics.json',M)
assert M['artifact_bytes']==sum(p.stat().st_size for p in H.rglob('*') if p.is_file())<1000000
assert M['CPU_wall_seconds']+V['CPU_wall_seconds']<120
print(json.dumps({'independent_verification':V,'total_artifact_bytes':M['artifact_bytes']},indent=2))
