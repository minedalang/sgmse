"""Independent CPU-only replay-artifact validation; no torch/model imports."""
from pathlib import Path
import io,json,hashlib
import numpy as np
import soundfile as sf
H=Path(__file__).resolve().parent;P=H.parent
D=json.loads((H/'metrics.json').read_text());C=D['config'];PROV=json.loads((H/'attempt_provenance.json').read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def ah(a):return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()
def encode(x):
 b=io.BytesIO();sf.write(b,x,16000,format='WAV',subtype='PCM_16');return b.getvalue()
assert D['status']=='passed' and not D['failures'] and D['child_exit_code']==0
assert (D['frontend_calls'],D['backend_calls'],D['nfe'],D['trace_calls'])==(1,1,30,30)
assert D['pre_hashes']==D['post_hashes']==C['source_locks']
# Independently rehash non-checkpoint sources; checkpoint rehashes are already actual pre/post worker reads.
for p,h in C['source_locks'].items():
 if p!=C['checkpoint']:assert sha(p)==h
for name,m in D['artifact_manifest'].items():assert (H/name).stat().st_size==m['bytes'] and sha(H/name)==m['sha256']
for name,h in PROV['parent_failure_artifact_sha256'].items():assert sha(P/name)==h
for name,h in PROV['frozen_files_copied_byte_identical'].items():assert sha(H/name)==sha(P/name)==h
assert sha(H/'frozen_manifest.json')==PROV['frozen_manifest_sha256']==sha(P/'frozen_manifest.json')
front=np.load(H/'outputs/frontend_raw.npy',allow_pickle=False);back=np.load(H/'outputs/backend_raw.npy',allow_pickle=False)
assert front.shape==(1,1,480000) and back.shape==(480000,) and front.dtype==back.dtype==np.float32
assert np.isfinite(front).all() and np.isfinite(back).all()
assert ah(front)==D['frontend_raw']['sha256']==D['backend_anchor']['sha256'];assert ah(back)==D['backend_raw']['sha256']
assert encode(front.reshape(-1))==Path(C['frontend_pcm']).read_bytes()
assert encode(np.clip(back,-1,1))==Path(C['off_pcm']).read_bytes()==(H/'outputs'/C['basename']).read_bytes()
full,sr=sf.read(H/'outputs'/C['basename'],dtype='int16');clip,csr=sf.read(H/'outputs/off_context_18p75_20p75.wav',dtype='int16')
historical_clip,hsr=sf.read(C['off_clip'],dtype='int16');assert sr==csr==hsr==16000 and np.array_equal(full[300000:332000],clip) and np.array_equal(clip,historical_clip)
assert sha(H/'outputs/off_context_18p75_20p75.wav')==sha(C['off_clip'])
noisy,nsr=sf.read(C['noisy'],dtype='float32');assert nsr==16000 and noisy.shape==(480000,) and ah(noisy)==D['backend_condition']['sha256']
assert float(np.max(np.abs(noisy)))==D['normalization_reference_peak']
seed=(2026+int.from_bytes(hashlib.blake2b(C['basename'].encode(),digest_size=4).digest(),'little'))%2**32;assert seed==C['effective_seed']==369124093
rng_count=0
for key,meta in D.items():
 if key.startswith('rng_') and isinstance(meta,dict):
  for device,m in meta.items():
   x=np.load(H/'outputs'/(key+'_'+device+'.npy'),allow_pickle=False);assert list(x.shape)==m['shape'] and str(x.dtype)==m['dtype'] and ah(x)==m['sha256'];rng_count+=1
for device,m in D['first_trace']['rng_post_predictor'].items():
 x=np.load(H/'outputs'/('rng_first_trace_post_predictor_'+device+'.npy'),allow_pickle=False);assert ah(x)==m['sha256'];rng_count+=1
assert D['rng_before_frontend']==D['rng_after_frontend']
assert D['first_trace']['step_index']==0 and D['first_trace']['t']==[1.0]
assert D['first_trace']['y']==D['sampler_settings']['y']
assert not any(D['sampler_calibrators'].values())
parent=json.loads((P/'metrics.json').read_text());assert parent['frontend_calls']==parent['backend_calls']==0
record={'status':'passed','method':'independent CPU artifact/serialization/source/RNG verification, no model calls','supervisor_status':'passed','attempts':2,'attempt01':'failed import,0front/0back; six parent failure artifacts unchanged','attempt02_frontend_calls':1,'attempt02_backend_calls':1,'total_frontend_calls':1,'total_backend_calls':1,'NFE':30,'trace_callbacks':30,'historical_PCM_and_byte_gates_recomputed':True,'historical_raw_equality_claim':False,'raw_frontend_float32_sha256':ah(front),'raw_backend_float32_sha256':ah(back),'source_lock_count':len(C['source_locks']),'checkpoint_hash_status':'actual pre/post rehash by replay; values verified against config, no third checkpoint read','supervisor_artifact_entries_verified':len(D['artifact_manifest']),'saved_rng_arrays_verified':rng_count,'frontend_rng_unchanged':True,'first_trace_x_t_semantics':'pre-predictor after identity none corrector','first_trace_rng_semantics':'post-predictor callback, distinct from before-sampler states','parent_failure_preserved':True,'raw_backend_clipping_samples':int(np.count_nonzero((back < -1)|(back > 1))),'source_noisy_peak':D['normalization_reference_peak'],'supervisor_wall_seconds':D['supervisor_wall_seconds'],'supervisor_budget_seconds':180,'artifact_cap_bytes':15000000}
(H/'artifact_validation.json').write_text(json.dumps(record,indent=2)+'\n');print(json.dumps(record,indent=2))
