"""Read-only validation of frozen inputs and cached outputs; no model forward."""
from pathlib import Path
import hashlib,json,shutil,sys,platform
import numpy as np
import torch
import torch.nn.functional as F
P=Path(__file__).resolve().parent

def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def ah(x): return hashlib.sha256(x.tobytes(order='C')).hexdigest()
def rms(x): return float(np.sqrt(np.mean(np.square(x,dtype=np.float64))))
f=json.loads((P/'inputs/recipe_frozen.json').read_text())
rows=json.loads((P/'recognition_outputs/per_sample.json').read_text())
assert len(rows)==144
assert sha(P/'inputs/recipe_frozen.json')=='8936e3edc3f337356cba7943745a9ed2a9939a76e7ce0da4afd21eb84749d769'
assert sha(P/'fixed_noise.py')==f['runner_sha256']
assert sha(P/'config.yaml')==f['config_sha256']
assert sha(f['source_manifest'])==f['source_manifest_sha256']
assert sha(f['prior_scoring_module'])==f['prior_scoring_module_sha256']
model=Path(f['model_path'])
assert {str(p.relative_to(model)):sha(p) for p in sorted(model.rglob('*')) if p.is_file()}==f['model_files_sha256']
caches={}
for r in f['clean']+f['noise']:
 assert sha(r.get('wav_path',r.get('source_path')))==r.get('sha256',r.get('source_sha256'))
 assert sha(P/r['array_path'])==r['array_sha256']
 x=np.load(P/r['array_path'],allow_pickle=False)['waveform']
 assert ah(x)==r['float64_sha256']
 caches[r.get('sample_id',r.get('noise_id'))]=x
snrs=[]
for r in f['recipes']:
 clean=caches[r['sample_id']]
 x=clean if r['noise_id'] is None else clean+r['noise_gain']*caches[r['noise_id']]
 x32=x.astype(np.float32)
 assert ah(x)==r['float64_sha256'] and ah(x32)==r['model_input_float32_sha256']
 if r['noise_id'] is not None:
  gain=.01*10**(-r['snr_db']/20)
  assert r['noise_gain']==gain
  noise=gain*caches[r['noise_id']]
  s64=20*np.log10(rms(clean)/rms(noise))
  # Float32 mixture and separately converted clean define effective residual noise.
  c32=clean.astype(np.float32).astype(np.float64)
  effective_noise=x32.astype(np.float64)-c32
  s32=20*np.log10(rms(c32)/rms(effective_noise))
  snrs.append({'condition':r['condition'],'sample_id':r['sample_id'],'target_snr_db':r['snr_db'],'actual_snr_db_float64':s64,'effective_snr_db_float32':s32,'float32_minus_float64_db':s32-s64})
max_ctc=max_prob=max_json=0.
for r in rows:
 path=P/r['array_path']; assert sha(path)==r['array_sha256']
 with np.load(path,allow_pickle=False) as a:
  logits=torch.from_numpy(a['logits']); logp=logits.log_softmax(-1)
  targets=torch.from_numpy(a['candidate_token_ids'])
  scores=-F.ctc_loss(logp[:,None,:].expand(-1,24,-1).contiguous(),targets,torch.full((24,),len(logits),dtype=torch.long),torch.full((24,),3,dtype=torch.long),blank=0,reduction='none',zero_infinity=False).numpy()
  assert a['logits'].dtype==np.float32 and len(a['candidate_labels'])==24
  max_ctc=max(max_ctc,float(np.max(np.abs(scores-a['candidate_loglikelihoods']))))
  max_prob=max(max_prob,float(np.max(np.abs(logp.exp().numpy()-a['probabilities']))))
  max_json=max(max_json,max(abs(float(score)-r['all24_loglikelihoods'][str(label)]) for label,score in zip(a['candidate_labels'],scores)))
assert max_ctc==max_prob==max_json==0.
conditions={}
for c in dict.fromkeys(r['condition'] for r in snrs):
 rr=[r for r in snrs if r['condition']==c]
 conditions[c]={'n':len(rr),'actual_float64_min_db':min(r['actual_snr_db_float64'] for r in rr),'actual_float64_max_db':max(r['actual_snr_db_float64'] for r in rr),'effective_float32_min_db':min(r['effective_snr_db_float32'] for r in rr),'effective_float32_max_db':max(r['effective_snr_db_float32'] for r in rr),'max_absolute_float32_minus_float64_db':max(abs(r['float32_minus_float64_db']) for r in rr)}
result={'status':'passed','recognizer_forwards_in_validation':0,'cached_arrays':len(rows),'all24_ctc_scores_verified':len(rows)*24,'max_ctc_absolute_error':max_ctc,'max_probability_absolute_error':max_prob,'max_json_score_absolute_error':max_json,'recipe_sha256':sha(P/'inputs/recipe_frozen.json'),'all_144_recipe_and_float32_input_hashes_verified':True,'source_protocol_config_runner_checkpoint_hashes_unchanged':True,'identical_fixed_noise_and_gain_across_all_16_identities_per_condition':True,'float32_snr_definition':'rms(clean.astype(float32).astype(float64)) divided by rms(mixture.astype(float32).astype(float64) - clean.astype(float32).astype(float64)); this includes mixture rounding residual','snr_by_condition':conditions,'snr_per_sample':snrs,'environment':{'python':sys.version,'platform':platform.platform(),'torch':torch.__version__,'cuda_build':torch.version.cuda,'cudnn':torch.backends.cudnn.version(),'gpu_name':torch.cuda.get_device_name(0)},'disk_free_bytes':shutil.disk_usage(P).free}
assert result['disk_free_bytes']>2_000_000_000
(P/'worker_validation.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
print(json.dumps({k:v for k,v in result.items() if k not in ('snr_per_sample',)},indent=2))
