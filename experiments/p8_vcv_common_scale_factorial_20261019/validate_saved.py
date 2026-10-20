"""Independent saved-artifact checks, without model forwards."""
from pathlib import Path
import hashlib, importlib.util, io, json
import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as F
HERE=Path(__file__).resolve().parent;ROOT=HERE.parent.parent
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def ah(x):return hashlib.sha256(x.tobytes()).hexdigest()
def load(p,name):
 s=importlib.util.spec_from_file_location(name,p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
d=json.loads((HERE/'metrics.json').read_text());cfg=json.loads((HERE/'config.yaml').read_text())
assert d['status']=='completed' and (d['frontend_forwards'],d['backend_forwards'],d['recognizer_forwards'])==(0,16,16)
assert all(sha(p)==h for p,h in d['sha256_locks'].items())
previous=json.loads((ROOT/cfg['previous_metrics']).read_text());priorbase=(ROOT/cfg['previous_metrics']).parent
checks=[]
for c in d['cases']:
 raw=np.load(HERE/c['raw_path'],allow_pickle=False);assert raw.shape==(32000,) and raw.dtype==np.float32 and np.isfinite(raw).all()
 assert sha(HERE/c['raw_path'])==c['raw_sha256'] and ah(raw)==c['float32_sha256'] and sha(HERE/c['wav_path'])==c['wav_sha256']
 pcm,sr=sf.read(HERE/c['wav_path'],dtype='int16');assert sr==16000 and pcm.shape==(32000,)
 encoded=io.BytesIO();sf.write(encoded,np.clip(raw,-1,1),16000,format='WAV',subtype='PCM_16');assert hashlib.sha256(encoded.getvalue()).hexdigest()==c['wav_sha256']
 assert c['nfe']==c['trace_steps']==30 and c['condition_normalized_peak']<=1 and not c['outside_native_norm_range']
 if c['anchor_origin']==c['noisy_donor']=='k':
  p=next(z for z in previous['cases'] if z['condition']==c['condition'] and z['anchor_id']==c['anchor_id'] and not z['cross'])
  assert np.array_equal(raw,np.load(priorbase/p['raw_path'],allow_pickle=False)) and c['raw_sha256']==p['raw_sha256'] and c['wav_sha256']==p['wav_sha256']
 s=c['ctc'];p=HERE/s['array_path'];assert sha(p)==s['array_sha256'] and 'top1_correct' not in s and 'binary_correct' not in s
 with np.load(p,allow_pickle=False) as z:
  logits=torch.from_numpy(z['logits'].copy());labels=z['candidate_labels'].tolist();target=torch.from_numpy(z['candidate_token_ids'].copy());n=len(logits)
  actual=-F.ctc_loss(logits.log_softmax(-1)[:,None,:].expand(-1,24,-1).contiguous(),target,torch.full((24,),n,dtype=torch.long),torch.full((24,),3,dtype=torch.long),blank=0,reduction='none').numpy()
  expected=np.asarray([s['all24_loglikelihoods'][x] for x in labels],dtype=np.float32)
  err=float(np.max(np.abs(actual-expected)));assert np.allclose(actual,expected,atol=1e-4,rtol=0)
  assert np.array_equal(z['candidate_loglikelihoods'],expected) and np.allclose(logits.softmax(-1).numpy(),z['probabilities'],atol=2e-6,rtol=0)
  assert abs(c['q_loglikelihood_g_minus_k']-float(expected[labels.index('g')]-expected[labels.index('k')]))<1e-5
  checks.append(dict(case_id=c['case_id'],candidate_scores_checked=24,max_abs_error=err))
fm=load(ROOT/cfg['recipe_helper'],'independent_recipe');old=json.loads((ROOT/cfg['original_recipe']).read_text());fresh=json.loads((ROOT/cfg['fresh_recipe']).read_text())
noises={r['noise_id']:np.load(fm.HERE/r['array_path'],allow_pickle=False)['waveform'] for r in old['noise']}
for group in d['groups']:
 cells=[c for c in d['cases'] if c['speaker']==group['speaker'] and c['condition']==group['condition']];assert len(cells)==4
 frozen=old if group['speaker']=='f6' else fresh;waves={};clean={};recipes={}
 for phone in ['g','k']:
  sid=group[phone+'_id'];row=next(r for r in frozen.get('clean',frozen.get('rows')) if r['sample_id']==sid)
  clean[phone]=np.load(fm.HERE/row['array_path'],allow_pickle=False)['waveform'] if group['speaker']=='f6' else np.load(ROOT/'experiments/p8_q_vcv_speaker_replication_20261019/outputs'/(sid+'_clean.npy'),allow_pickle=False)
  rec=next(r for r in frozen['recipes'] if r['sample_id']==sid and r['condition']==group['condition']);recipes[phone]=rec
  waves[phone]=fm.recipe_wave(rec,dict(noises,**{sid:clean[phone]}))
 ref=max(float(np.abs(waves[p]).max()) for p in ['g','k']);assert ref==group['common_ref']
 assert (recipes['g']['noise_id'],recipes['g']['noise_gain'])==(recipes['k']['noise_id'],recipes['k']['noise_gain'])
 normalized_noise=recipes['g']['noise_gain']*noises[recipes['g']['noise_id']]/ref;assert ah(normalized_noise)==group['normalized_noise_sha256']
 residual=waves['g'].astype(np.float64)-waves['k'].astype(np.float64)-clean['g']+clean['k'];assert np.abs(residual).max()<=group['float32_error_bound']
 assert len({c['common_ref'] for c in cells})==len({c['normalized_noise_sha256'] for c in cells})==1
 for ap in ['g','k']:
  pair=[c for c in cells if c['anchor_origin']==ap]
  cached=next(z for z in previous['frontend_replays'] if z['condition']==group['condition'] and z['sample_id']==group[ap+'_id'])
  assert all(c['anchor_raw_sha256']==ah(np.load(priorbase/cached['raw_path'],allow_pickle=False)) for c in pair)
  for key in ['normalized_anchor_sha256','cpu_rng_sha256','cuda_rng_sha256','initial_xT_sha256','regeneration_strength','common_ref','seed']:
   assert pair[0][key]==pair[1][key]
  for c in pair:
   expected_peak=float(np.abs(waves[c['noisy_donor']]/ref).max())
   assert abs(c['condition_normalized_peak']-expected_peak)<=np.finfo(np.float32).eps and expected_peak<=1
 q={c['anchor_origin']+c['noisy_donor']:c['q_loglikelihood_g_minus_k'] for c in cells}
 assert q==group['q']
 assert group['anchor_effect']==((q['gg']-q['kg'])+(q['gk']-q['kk']))/2
 assert group['condition_effect']==((q['gg']-q['gk'])+(q['kg']-q['kk']))/2
 assert group['interaction']==(q['gg']-q['gk'])-(q['kg']-q['kk'])
result=dict(status='passed',no_new_model_forwards=True,audio_pairs_validated=16,ctc_candidate_scores_recomputed=384,ctc_checks=checks,source_locks_validated=len(d['sha256_locks']),exact_native_k_gates_validated=4,common_scale_noise_groups_validated=4,held_row_geometry_checks=8,factorial_contrasts_validated=12)
(HERE/'outputs/independent_validation.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k!='ctc_checks'}))
