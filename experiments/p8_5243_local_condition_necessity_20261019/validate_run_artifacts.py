"""Independent CPU verification of saved frozen local-diagnostic artifacts."""
from pathlib import Path
import json,hashlib,io
import numpy as np
import soundfile as sf
H=Path(__file__).resolve().parent
def read(p):return json.loads(Path(p).read_text())
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
def ah(x):return hashlib.sha256(np.ascontiguousarray(x).tobytes()).hexdigest()
def descriptor(x,r):assert x.shape==tuple(r['shape']) and str(x.dtype)==r['dtype'] and ah(x)==r['sha256']
def pcm(x):
 b=io.BytesIO();sf.write(b,x,16000,format='WAV',subtype='PCM_16');return b.getvalue()
D=read(H/'metrics.json');C=D['config'];S=read(H/'selection_frozen.json');B=read(C['baseline_metrics'])
assert D['status']=='completed' and not D['failures'] and D['calls']=={'frontend':0,'backend':4,'recognizer':4}
assert D['native_backend_gate_passed'] and D['native_measurement_gate']['passed'] and D['backend_released_before_classifier']
assert D['pre_source_hashes']==D['post_source_hashes']==C['source_locks']
weightpaths=[]
for p,h in C['source_locks'].items():
 if Path(p).suffix in ['.ckpt','.bin','.safetensors']:weightpaths.append(p)
 else:assert sha(p)==h,p
for n,h in read(H/'frozen_execution_manifest.json').items():assert sha(H/n)==h,n
for n,m in D['artifact_manifest'].items():assert (H/n).stat().st_size==m['bytes'] and sha(H/n)==m['sha256'],n
noisy,sr=sf.read(C['noisy'],dtype='float32');clean,cs=sf.read(C['clean'],dtype='float64');noise,ns=sf.read(C['noise'],dtype='float64');assert sr==cs==ns==16000
anchor=np.load(C['anchor'],allow_pickle=False).reshape(-1);base_raw=np.load(C['baseline_raw'],allow_pickle=False)
expected_rows={'target':S['target'],**S['selected_controls']};w=np.ones(3200);r=.5-.5*np.cos(np.linspace(0,np.pi,320));w[:320]=r;w[-320:]=r[::-1]
conditions={'native':noisy};regions={k:(v['start_sample'],v['end_sample_exclusive']) for k,v in expected_rows.items()}
for name,row in expected_rows.items():
 a,b=regions[name];alpha=row.get('alpha',row.get('alpha_if_energy_eligible'));ideal=alpha*w*clean[a:b];x=noisy.copy();x[a:b]=(noisy[a:b].astype(np.float64)-ideal).astype(np.float32);actual=noisy[a:b].astype(np.float64)-x[a:b];g=D['input_guards'][name]
 for key,value in [('mask',w),('ideal_removed_clean',ideal),('actual_condition_delta',actual)]:assert np.array_equal(np.load(H/'outputs'/(name+'_'+key+'.npy'),allow_pickle=False),value)
 assert np.array_equal(x[:a],noisy[:a]) and np.array_equal(x[b:],noisy[b:]);assert np.abs(x).max().tobytes()==np.abs(noisy).max().tobytes();assert np.isfinite(x).all();assert ah(x)==g['condition_sha256']==C['expected_conditions'][name]['condition_float32_sha256'];assert abs(float(actual@actual)-S['target']['weighted_clean_energy'])/S['target']['weighted_clean_energy']<1e-5
 oldres=noisy[a:b].astype(np.float64)-clean[a:b]-noise[a:b];newres=x[a:b].astype(np.float64)-(clean[a:b]-ideal)-noise[a:b];assert float(np.abs(newres-oldres).max())==g['noise_residual_change_max']<=g['float32_roundoff_bound'];assert float(np.abs(newres).max())==g['mixture_residual_max']<=g['mixture_bound'];conditions[name]=x
raws={n:np.load(H/'outputs'/(n+'_raw.npy'),allow_pickle=False) for n in D['variants']}
assert np.array_equal(raws['native'],base_raw)
cache=np.load(C['cached_probabilities'],allow_pickle=False);old=cache[C['cache_probabilities_key']];ids=cache[C['cache_token_ids_key']];oldarg=cache[C['cache_argmax_key']]
vocab=read(Path(C['classifier_model'])/'vocab.json');id2={i:t for t,i in vocab.items()};checks={};alloutside=np.ones(480000,bool)
for a,b in regions.values():alloutside[a:b]=False
for name,v in D['variants'].items():
 x=raws[name];assert x.shape==(480000,) and x.dtype==np.float32 and np.isfinite(x).all();descriptor(x,v['raw_output']);assert pcm(np.clip(x,-1,1))==(H/'outputs'/(name+'.wav')).read_bytes()
 full,fs=sf.read(H/'outputs'/(name+'.wav'),dtype='int16');clip,cs=sf.read(H/'outputs'/(name+'_context.wav'),dtype='int16');assert fs==cs==16000 and np.array_equal(full[300000:332000],clip);assert pcm(clip)==(H/'outputs'/(name+'_context.wav')).read_bytes()
 descriptor(anchor,v['anchor']);descriptor(conditions[name],v['condition']);assert v['reference_peak']==float(np.abs(noisy).max());assert v['rng_before_backend']==B['rng_before_backend'];assert v['rng_before_sampler']==B['rng_before_sampler'];assert v['rng_after_backend']==B['rng_after_backend'];assert v['normalized_anchor']==B['first_trace']['y'];assert v['nfe']==v['trace_calls']==30
 for k in ['step_index','t','stepsize','x_t','rng_post_predictor']:assert v['first_trace'][k]==B['first_trace'][k]
 assert v['sampler_settings']==D['variants']['native']['sampler_settings'] and v['enhance_settings']==C['enhance_settings']
 logits=np.load(H/'outputs'/(name+'_ctc_logits.npy'),allow_pickle=False);probs=np.load(H/'outputs'/(name+'_ctc_probabilities.npy'),allow_pickle=False);greedy=np.load(H/'outputs'/(name+'_ctc_greedy_ids.npy'),allow_pickle=False)
 descriptor(logits,v['logits']);descriptor(probs,v['probs']);descriptor(greedy,v['greedy_ids']);assert logits.shape==probs.shape==(99,392) and np.isfinite(logits).all();e=np.exp(logits.astype(np.float64)-logits.max(-1,keepdims=True));recomputed=e/e.sum(-1,keepdims=True);softmaxerr=float(np.max(np.abs(recomputed-probs)));assert softmaxerr<1e-6 and np.array_equal(greedy,probs.argmax(-1))
 scores=v['fixed_d_metrics'];p=probs[:,C['d_token_id']];assert scores['fixed_frames']==[35,36] and scores['neighborhood_frames']==[25,46];assert scores['fixed_posterior_mean']==float(p[35:36].mean());assert scores['neighborhood_posterior_max']==float(p[25:46].max());assert scores['neighborhood_posterior_mean']==float(p[25:46].mean());assert scores['neighborhood_peak_frame']==25+int(p[25:46].argmax());assert scores['neighborhood_d_top1_frames']==[i for i in range(25,46) if greedy[i]==C['d_token_id']]
 if name=='native':assert np.max(np.abs(probs[:,ids]-old))<=1e-5 and np.array_equal(greedy,oldarg)
 fclip,_=sf.read(H/'outputs'/(name+'_context.wav'),dtype='float32');mp=v['measurement_processing']['clip_pcm_float32'];descriptor(fclip,mp['descriptor']);assert mp['mean']==float(fclip.astype(np.float64).mean()) and mp['rms']==float(np.sqrt(np.mean(fclip.astype(np.float64)**2))) and mp['std']==float(fclip.astype(np.float64).std())
 diff=x.astype(np.float64)-raws['native'].astype(np.float64)
 for key,(a,b) in regions.items():assert v['output_delta_vs_native'][key]['rms']==float(np.sqrt(np.mean(diff[a:b]**2))) and v['output_delta_vs_native'][key]['max_abs']==float(np.max(np.abs(diff[a:b])))
 assert v['output_delta_vs_native']['outside_all_three_windows']['rms']==float(np.sqrt(np.mean(diff[alloutside]**2)))
 collapsed=[];last=None
 for g in greedy:
  if int(g)!=last and int(g)!=vocab['<pad>']:collapsed.append(id2[int(g)])
  last=int(g)
 checks[name]={'softmax_max_abs_error':softmaxerr,'fixed_d':scores['fixed_posterior_mean'],'nearby_max_d':scores['neighborhood_posterior_max'],'nearby_mean_d':scores['neighborhood_posterior_mean'],'nearby_peak_frame':scores['neighborhood_peak_frame'],'independent_greedy_collapsed_tokens':collapsed,'all_checks_passed':True}
assert sha(H/'outputs/native.wav')==sha(C['historical_off']) and sha(H/'outputs/native_context.wav')==sha(C['historical_clip'])
result={'status':'passed','no_model_calls':True,'source_locks':len(C['source_locks']),'nonweight_sources_independently_rehashed':len(C['source_locks'])-len(weightpaths),'weights_with_actual_runtime_pre_post_hashes_verified_not_reread':weightpaths,'frozen_execution_files':len(read(H/'frozen_execution_manifest.json')),'supervisor_artifact_manifest_entries_verified':len(D['artifact_manifest']),'checks':checks,'input_guards_reconstructed':True,'initial_state_RNG_anchor_reference_guards_verified':True,'cached_native_target_probabilities_and99argmax_verified':True,'native_raw_and_historical_PCM_byte_gates_recomputed':True}
(H/'artifact_validation.json').write_text(json.dumps(result,indent=2,ensure_ascii=False)+'\n');print(json.dumps(result,indent=2,ensure_ascii=False))
