"""Validate fresh-speaker caches only; no enhancement/recognizer forwards."""
from pathlib import Path
import hashlib,json,importlib.util,io,shutil,sys,platform
import numpy as np
import torch
import torch.nn.functional as F
import soundfile as sf
from scipy.signal import resample_poly
P=Path(__file__).resolve().parent;ROOT=P.parent.parent;PREV=P.parent/'p8_q_vcv_fixed_noise_20261018'
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
def ah(x):return hashlib.sha256(x.tobytes()).hexdigest()
def load(p,n):
 s=importlib.util.spec_from_file_location(n,p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
def rms(x):return float(np.sqrt(np.mean(x*x,dtype=np.float64)))
fm=load(PREV/'fixed_noise.py','fixed');helper=load(P.parent/'p8_q_vcv_enhance_follow_20261018/run_follow.py','helper')
m=json.loads((P/'metrics.json').read_text());f=json.loads((P/'outputs/recipe_frozen.json').read_text());rr=json.loads((P/'outputs/per_sample.json').read_text());prior=json.loads((PREV/'inputs/recipe_frozen.json').read_text());manifest=json.loads((P/'retrieval/selected_manifest_frozen.json').read_text())
assert m['status'] in ('completed','stopped_clean_gate_failed')
assert sha(P/'retrieval/selected_manifest_frozen.json')=='31c4e4aa9abd9a2cae0cf8b2576942506e3f1cb18cfdb5d7ffd356ff24773ff2'
assert all(sha(p)==digest for p,digest in m['source_sha256'].items())
assert {r['speaker'] for r in f['rows']}=={'f18','m5','f20','m15'}
assert not {r['speaker'] for r in f['rows']}&set(manifest['excluded_previous28_speakers'])
caches={};rows={r['sample_id']:r for r in f['rows']}
for sid,r in rows.items():
 w,sr=sf.read(r['wav_path'],dtype='float64',always_2d=True);assert sr==25000 and w.shape[1]==1
 z=resample_poly(w[:,0],16,25);x=np.zeros(32000,dtype=np.float64);x[8000:8000+len(z)]=z;x*=.01/rms(x)
 saved=np.load(P/'outputs'/(sid+'_clean.npy'),allow_pickle=False);assert np.array_equal(x,saved) and r['resampled_samples']==len(z)
 assert np.count_nonzero(x[:8000])==0 and np.count_nonzero(x[8000+len(z):])==0
 caches[sid]=x
for r in prior['noise']:
 assert sha(r['source_path'])==r['source_sha256']
 path=PREV/r['array_path'];assert sha(path)==r['array_sha256'];x=np.load(path,allow_pickle=False)['waveform'];assert ah(x)==r['float64_sha256'];caches[r['noise_id']]=x
support=[];wave_hashes={}
for r in f['recipes']:
 sid=r['sample_id'];x=caches[sid] if r['noise_id'] is None else caches[sid]+r['noise_gain']*caches[r['noise_id']]
 assert ah(x)==r['float64_sha256'] and ah(x.astype(np.float32))==r['model_input_float32_sha256'];wave_hashes[(r['condition'],sid)]=ah(x.astype(np.float32))
 if r['noise_id'] is not None:
  assert r['noise_gain']==float(.01*10**(-r['snr_db']/20));n=r['noise_gain']*caches[r['noise_id']];c=caches[sid];end=8000+rows[sid]['resampled_samples']
  carrier=20*np.log10(rms(c)/rms(n));s=10*np.log10(np.mean(c[8000:end]**2)/np.mean(n[8000:end]**2));c32=c.astype(np.float32).astype(np.float64);x32=x.astype(np.float32).astype(np.float64);effective32=20*np.log10(rms(c32)/rms(x32-c32))
  assert abs(carrier-r['snr_db'])<1e-12
  support.append({'condition':r['condition'],'sample_id':sid,'carrier_snr64_db':float(carrier),'effective_input32_snr_db':float(effective32),'source_recording_support_snr_db':float(s),'recording_duration_seconds':rows[sid]['resampled_samples']/16000})
for r in rr:
 path=P/r['array_path'];assert sha(path)==r['array_sha256'];a=np.load(path,allow_pickle=False);logits=torch.from_numpy(a['logits']);logp=logits.log_softmax(-1)
 scores=-F.ctc_loss(logp[:,None,:].expand(-1,24,-1).contiguous(),torch.from_numpy(a['candidate_token_ids']),torch.full((24,),len(logits),dtype=torch.long),torch.full((24,),3,dtype=torch.long),blank=0,reduction='none',zero_infinity=False).numpy()
 assert np.array_equal(scores,a['candidate_loglikelihoods']) and np.array_equal(logp.exp().numpy(),a['probabilities'])
 assert dict(zip(map(str,a['candidate_labels']),map(float,scores)))==r['all24_loglikelihoods']
 rank=sorted(r['all24_loglikelihoods'].items(),key=lambda x:-x[1]);assert r['top1_prediction']==rank[0][0] and r['top1_correct']==(rank[0][0]==r['consonant'])
 assert r['binary_correct']==(r['binary_loglikelihood_margin']>0)
for c,arms in m['summaries'].items():
 for arm,s in arms.items():
  name='baseline' if arm in ('clean','noisy') else arm;rs=[r for r in rr if r['condition']==c and r['arm']==name]
  expected=fm.summarize(rs,f['pairs']);assert expected=={k:v for k,v in s.items() if k!='binary_correct_but_24way_wrong'}
  if name!='baseline':assert s['binary_correct_but_24way_wrong']==[r['sample_id'] for r in rs if r['binary_correct'] and not r['top1_correct']]
gate=fm.eligible(m['summaries']['padded_clean']['clean'],pair_gate=False);assert gate==m['clean_gate_passed']
audios=0;max_quant=0.;clipping=0;comparisons={};stability={}
if gate:
 assert (m['frontend_forwards'],m['backend_forwards'],m['recognizer_forwards'])==(64,128,272) and len(rr)==272
 aa=json.loads((P/'outputs/audio_manifest.json').read_text());assert len(aa)==320
 for a in aa:
  x=np.load(P/a['raw_path'],allow_pickle=False);assert x.shape==(32000,) and x.dtype==np.float32 and np.isfinite(x).all();assert sha(P/a['raw_path'])==a['raw_sha256'] and ah(x)==a['float32_sha256'];assert sha(P/a['wav_path'])==a['wav_sha256']
  buf=io.BytesIO();sf.write(buf,np.clip(x,-1,1),16000,format='WAV',subtype='PCM_16');assert buf.getvalue()==(P/a['wav_path']).read_bytes();w,sr=sf.read(P/a['wav_path']);assert sr==16000;max_quant=max(max_quant,float(np.max(np.abs(w-np.clip(x.astype(np.float64),-1,1)))))
  assert helper.quiet(x,rows[a['sample_id']])==a['quiet_padding'];count=int(np.count_nonzero((x<-1)|(x>=1)));assert count==a['pcm_clipping_count'];clipping+=count
  if a['arm']=='noisy':assert ah(x)==wave_hashes[(a['condition'],a['sample_id'])]
  if a['arm'] in helper.ARMS:assert a['seed']==helper.seed(rows[a['sample_id']]['primary_pair_ids'][0])
  if a['arm']=='y_pre':assert a['frontend_rng_unchanged']
  audios+=1
 for sm in f['seed_matrix']:assert sm['seed']==helper.seed(sm['pair_id']) and all(s==sm['seed'] for s in sm['arms'].values())
 assert len({s['seed'] for s in f['seed_matrix']})==8
 for c in [r for r in m['summaries'] if r!='padded_clean']:
  assert m['noisy_input_eligibility'][c]==fm.eligible(m['summaries'][c]['noisy'])
  indexed={arm:{r['sample_id']:r for r in rr if r['condition']==c and r['arm']==arm} for arm in ['baseline']+helper.ARMS};comparisons[c]={}
  for before,after in [('baseline','y_pre'),('baseline','off'),('baseline','router'),('y_pre','off'),('off','router')]:
   b=indexed[before];a=indexed[after];tr={}
   for metric in ('top1_correct','binary_correct'):
    tr[metric]={'correct_to_wrong':sum(b[s][metric] and not a[s][metric] for s in a),'wrong_to_correct':sum(not b[s][metric] and a[s][metric] for s in a)}
    if before in ('baseline','y_pre') and after in helper.ARMS:assert tr[metric]==m['transitions'][c][after]['noisy' if before=='baseline' else before][metric]
   comparisons[c][after+'_versus_'+before]=tr
 for snr in [-6,6]:
  stability[str(snr)]={}
  for arm in ['baseline']+helper.ARMS:
   a={r['sample_id']:r for r in rr if r['condition']==f'N1_snr{snr:+d}' and r['arm']==arm};b={r['sample_id']:r for r in rr if r['condition']==f'N2_snr{snr:+d}' and r['arm']==arm};stability[str(snr)][arm]={}
   for metric,pred in [('24way','top1_prediction'),('binary','binary_prediction')]:
    correct='top1_correct' if metric=='24way' else 'binary_correct';stability[str(snr)][arm][metric]={'n':16,'changed_prediction':sum(a[s][pred]!=b[s][pred] for s in a),'both_correct':sum(a[s][correct] and b[s][correct] for s in a),'stable_incorrect':sum(a[s][pred] is not None and a[s][pred]==b[s][pred] and not a[s][correct] for s in a)}
   expected=dict(m['noise_instability'][str(snr)][arm]);expected['changed_prediction']=expected.pop('prediction_changed');assert stability[str(snr)][arm]['24way']==expected
else:assert (m['frontend_forwards'],m['backend_forwards'],m['recognizer_forwards'])==(0,0,16) and len(rr)==16
support_summary={}
for c in dict.fromkeys(r['condition'] for r in support):
 rs=[r for r in support if r['condition']==c];x=[r['source_recording_support_snr_db'] for r in rs];support_summary[c]={'n':len(x),'minimum_db':min(x),'mean_db':float(np.mean(x)),'maximum_db':max(x),'carrier_snr_db':rs[0]['carrier_snr64_db']}
v={'status':'passed','model_forwards':0,'clean_gate_passed':gate,'scores_verified':len(rr)*24,'max_ctc_probability_json_error':0.,'raw_and_pcm16_audio_verified':audios,'source_checkpoint_protocol_locks_unchanged':True,'frozen_manifest_sha256':sha(P/'retrieval/selected_manifest_frozen.json'),'all_80_clean_noisy_recipes_verified':True,'matched_noise_and_gain_per_pair':True,'actual_common_pair_seeds_verified':gate,'frontend_rng_invariance_verified':gate,'pcm16_quantization_max_error':max_quant,'pcm_clipping_count_total':clipping,'arm_comparisons':comparisons,'noise_stability':stability,'support_snr_summary':support_summary,'support_snr_per_sample':support,'support_definition':'entire resampled source recording [8000,8000+resampled_samples), not phoneme-local or clean-active','environment':{'python':sys.version,'platform':platform.platform(),'torch':torch.__version__,'cuda_build':torch.version.cuda,'cudnn':torch.backends.cudnn.version(),'gpu':torch.cuda.get_device_name(0)},'disk_free_bytes':shutil.disk_usage(P).free}
assert v['disk_free_bytes']>2_000_000_000
(P/'worker_validation.json').write_text(json.dumps(v,indent=2,allow_nan=False)+'\n');print(json.dumps({k:v for k,v in v.items() if k not in ('support_snr_per_sample',)},indent=2))
