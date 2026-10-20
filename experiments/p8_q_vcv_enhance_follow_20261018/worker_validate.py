"""Verify cached follow-up artifacts; never runs enhancement or recognizer inference."""
from pathlib import Path
import hashlib,json,importlib.util,shutil,platform,sys
import io
import numpy as np
import torch
import torch.nn.functional as F
import soundfile as sf
P=Path(__file__).resolve().parent; PREV=P.parent/'p8_q_vcv_fixed_noise_20261018'
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
def ah(x):return hashlib.sha256(x.tobytes()).hexdigest()
def rms(x):return float(np.sqrt(np.mean(np.square(x,dtype=np.float64))))
def mod(path,name):
 s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
runner=mod(P/'run_follow.py','follow');fm=mod(PREV/'fixed_noise.py','frozen')
m=json.loads((P/'metrics.json').read_text());assert m['status']=='completed'
f=json.loads((PREV/'inputs/recipe_frozen.json').read_text());rr=json.loads((P/'outputs/per_sample.json').read_text());aa=json.loads((P/'outputs/audio_manifest.json').read_text());baseline=json.loads((PREV/'recognition_outputs/per_sample.json').read_text())
assert (m['frontend_forwards'],m['backend_forwards'],m['recognizer_forwards'])==(32,64,96)
assert all(sha(p)==h for p,h in m['readonly_sha256'].items())
assert len(rr)==96 and len(aa)==160
rows={r['sample_id']:r for r in f['clean']}; caches={r.get('sample_id',r.get('noise_id')):np.load(PREV/r['array_path'],allow_pickle=False)['waveform'] for r in f['clean']+f['noise']}
audio={(a['condition'],a['sample_id'],a['arm']):a for a in aa}
max_pcm=0.;max_energy=0.;clipping=0;pad_zero=True;min_raw_rms=1.;snrs=[]
for a in aa:
 raw=np.load(P/a['raw_path'],allow_pickle=False);assert raw.dtype==np.float32 and raw.shape==(32000,) and np.isfinite(raw).all()
 assert sha(P/a['raw_path'])==a['raw_sha256'] and ah(raw)==a['float32_sha256'] and sha(P/a['wav_path'])==a['wav_sha256']
 wav,sr=sf.read(P/a['wav_path'],dtype='float64');assert sr==16000 and wav.shape==(32000,) and sf.info(P/a['wav_path']).subtype=='PCM_16'
 # Reproduce exact installed libsndfile serialization, including its rounding.
 buf=io.BytesIO();sf.write(buf,np.clip(raw,-1,1),16000,format='WAV',subtype='PCM_16')
 assert buf.getvalue()==(P/a['wav_path']).read_bytes()
 err=float(np.max(np.abs(wav-np.clip(raw.astype(np.float64),-1,1))));max_pcm=max(max_pcm,err);assert err<=1/32768
 count=int(np.count_nonzero((raw < -1)|(raw >=1)));assert count==a['pcm_clipping_count'];clipping+=count
 q=runner.quiet(raw,rows[a['sample_id']])
 assert q==a['quiet_padding']
 if a['arm']=='clean':
  assert np.count_nonzero(raw[:4800])==0 and np.count_nonzero(raw[20800:])==0
 if a['arm'] in runner.ARMS:assert a['seed']==runner.seed(rows[a['sample_id']]['primary_pair_ids'][0])
 if a['arm']=='y_pre':assert a['frontend_rng_unchanged'] is True
for r in f['recipes']:
 if r['condition'] not in runner.CONDITIONS:continue
 x=fm.recipe_wave(r,caches);raw=np.load(P/audio[(r['condition'],r['sample_id'],'noisy')]['raw_path'])
 assert np.array_equal(x,raw)
 clean=caches[r['sample_id']];noise=r['noise_gain']*caches[r['noise_id']]
 assert r['noise_gain']==.01*10**(-6/20)
 snrs.append(20*np.log10(rms(clean)/rms(noise)))
for sm in m['seed_matrix']:
 assert sm['seed']==runner.seed(sm['pair_id']) and all(v==sm['seed'] for v in sm['arms'].values())
assert len({s['seed'] for s in m['seed_matrix']})==8
maxctc=maxprob=0.
# Recompute enhanced scores and every reused baseline from logits; no forward.
for r,root in [(r,P) for r in rr]+[(r,PREV) for r in baseline if r['condition'] in runner.CONDITIONS+['padded_clean']]:
 assert sha(root/r['array_path'])==r['array_sha256']
 a=np.load(root/r['array_path'],allow_pickle=False); logits=torch.from_numpy(a['logits']);logp=logits.log_softmax(-1)
 scores=-F.ctc_loss(logp[:,None,:].expand(-1,24,-1).contiguous(),torch.from_numpy(a['candidate_token_ids']),torch.full((24,),len(logits),dtype=torch.long),torch.full((24,),3,dtype=torch.long),blank=0,reduction='none',zero_infinity=False).numpy()
 assert np.array_equal(scores,a['candidate_loglikelihoods'])
 assert dict(zip(map(str,a['candidate_labels']),map(float,scores)))==r['all24_loglikelihoods']
 assert np.array_equal(logp.exp().numpy(),a['probabilities'])
 pred=max(r['all24_loglikelihoods'],key=r['all24_loglikelihoods'].get);assert pred==r['top1_prediction']
 assert r['top1_correct']==(pred==r['consonant']) and r['binary_correct']==(r['binary_loglikelihood_margin']>0)
for c,arms in m['summaries'].items():
 for arm,s in arms.items():
  if arm=='clean_reused': rs=[r for r in baseline if r['condition']=='padded_clean']
  elif arm=='noisy_reused':rs=[r for r in baseline if r['condition']==c]
  else:rs=[r for r in rr if r['condition']==c and r['arm']==arm]
  assert fm.summarize(rs,f['pairs'])==s
comparisons={};stability={}
for c in runner.CONDITIONS:
 comparisons[c]={}
 for before,after in [('y_pre','off'),('off','router')]:
  b={r['sample_id']:r for r in rr if r['condition']==c and r['arm']==before}; a={r['sample_id']:r for r in rr if r['condition']==c and r['arm']==after}
  comparisons[c][after+'_versus_'+before]={}
  for metric in ('top1_correct','binary_correct'):
   comparisons[c][after+'_versus_'+before][metric]={'correct_to_wrong':sum(b[s][metric] and not a[s][metric] for s in b),'wrong_to_correct':sum(not b[s][metric] and a[s][metric] for s in b),'correct_to_correct':sum(b[s][metric] and a[s][metric] for s in b),'wrong_to_wrong':sum(not b[s][metric] and not a[s][metric] for s in b)}
for arm in runner.ARMS:
 a={r['sample_id']:r for r in rr if r['condition']==runner.CONDITIONS[0] and r['arm']==arm};b={r['sample_id']:r for r in rr if r['condition']==runner.CONDITIONS[1] and r['arm']==arm}
 stability[arm]={}
 for metric,pred in [('24way','top1_prediction'),('binary','binary_prediction')]:
  correct='top1_correct' if metric=='24way' else 'binary_correct'
  stability[arm][metric]={'n':16,'same_prediction':sum(a[s][pred] is not None and a[s][pred]==b[s][pred] for s in a),'both_correct':sum(a[s][correct] and b[s][correct] for s in a),'stable_incorrect':sum(a[s][pred] is not None and a[s][pred]==b[s][pred] and not a[s][correct] for s in a)}
v={'status':'passed','enhancement_and_recognizer_forwards':0,'raw_audio_arrays_verified':160,'regenerated_input_hashes_verified':32,'enhanced_cached_ctc_scores_verified':96*24,'reused_baseline_cached_ctc_scores_verified':48*24,'max_ctc_score_error':0.,'max_probability_error':0.,'source_checkpoint_protocol_unchanged':True,'matched_noise_waveform_and_gain_per_pair':True,'shared_8_pair_seeds_across_both_identities_noises_arms':True,'all_32_frontends_rng_unchanged':True,'true_clean_padding_zeros_verified':True,'quiet_padding_stats_exact':True,'actual_snr64_min_db':min(snrs),'actual_snr64_max_db':max(snrs),'pcm16_conversion_max_error':max_pcm,'pcm_clipping_count_total':clipping,'pcm16_quantization_step':1/32768,'pcm16_one_lsb_dbfs':20*np.log10(1/32768),'padding_db_floor_policy':'Store linear RMS/mean-square. Exact zero dBFS is negative infinity; no finite surrogate floor inserted. PCM16 listening quantization may produce zero for very small raw samples. Primary scores use unclipped raw float32.','arm_comparisons':comparisons,'noise_stability':stability,'environment':{'python':sys.version,'platform':platform.platform(),'torch':torch.__version__,'cuda_build':torch.version.cuda,'cudnn':torch.backends.cudnn.version(),'gpu':torch.cuda.get_device_name(0)},'disk_free_bytes':shutil.disk_usage(P).free}
assert v['disk_free_bytes']>2_000_000_000
(P/'worker_validation.json').write_text(json.dumps(v,indent=2,allow_nan=False)+'\n');print(json.dumps(v,indent=2))
