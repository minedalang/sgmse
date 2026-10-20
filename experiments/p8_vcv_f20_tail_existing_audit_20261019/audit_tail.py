"""Existing case5 artifact audit: NumPy/SoundFile CPU only, no model calls."""
from pathlib import Path
import json,hashlib,time,io,csv,subprocess
import numpy as np
import soundfile as sf
H=Path(__file__).resolve().parent;R=H.parent.parent;T=time.monotonic();A=R/'experiments/p8_vcv_speech_component_ablation_20261019';F=R/'experiments/p8_q_vcv_speaker_replication_20261019';CID='fresh__N2_snr-6__f20aka1';SR=16000
assert not (H/'metrics.json').exists(),'No overwrite'
def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def ah(x):return hashlib.sha256(np.ascontiguousarray(x).tobytes()).hexdigest()
def dump(n,o):(H/n).write_text(json.dumps(o,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
D=read(A/'metrics.json');C=next(c for c in read(A/'selection_frozen.json')['cases'] if c['case_id']==CID);native=next(r for r in D['cases'] if r['case_id']==CID and not r['ablated']);ablate=next(r for r in D['cases'] if r['case_id']==CID and r['ablated']);recipe=read(F/'outputs/recipe_frozen.json');row=next(r for r in recipe['rows'] if r['sample_id']=='f20aka1');rec=next(r for r in recipe['recipes'] if r['sample_id']=='f20aka1' and r['condition']=='N2_snr-6');proof=next(p for p in read(A/'outputs/source_proofs.json') if p['case_id']==CID)
start=row['source_onset_sample'];end=start+row['resampled_samples'];intervals={'after_declared_support':[end,32000],'after_declared_support_plus100ms':[end+1600,32000]}
config={'case_id':CID,'CPU_only':True,'inference':False,'training':False,'downloads':False,'audio_output':False,'budget_seconds':120,'artifact_cap_bytes':2000000,'clean_support_samples':[start,end],'intervals':intervals,'disjoint_frame_samples':320,'highband_hz':[4000,7500],'window':'rectangular no overlap','user_sound_timing':'Only末尾 reported; no precise onset inferred'};dump('config.yaml',config)
git_before=subprocess.check_output(['git','status','--porcelain'],cwd=R).decode();(H/'git_status_before.txt').write_text(git_before)
# Scoring model configuration is read as JSON; no model initialization.
oldrecipe=read(R/'experiments/p8_q_vcv_fixed_noise_20261018/inputs/recipe_frozen.json');locks=read(A/'config.yaml')['sha256_locks'];vocabpath=next(Path(p) for p in locks if p.endswith('/vocab.json'));modelcfg=vocabpath.with_name('config.json');vocab=read(vocabpath);mconfig=read(modelcfg)
arrays={'clean':np.load(F/'outputs/f20aka1_clean.npy',allow_pickle=False),'y_pre':np.load(F/C['frontend_audio']['raw_path'],allow_pickle=False),'native_off':np.load(A/native['raw_path'],allow_pickle=False),'ablated_off':np.load(A/ablate['raw_path'],allow_pickle=False)}
wavepaths={'clean':F/'outputs/f20aka1_clean.npy','y_pre':F/C['frontend_audio']['raw_path'],'native_off':A/native['raw_path'],'ablated_off':A/ablate['raw_path']}
noise_rec=next(r for r in oldrecipe['noise'] if r['noise_id']=='N2');noise_path=R/'experiments/p8_q_vcv_fixed_noise_20261018'/noise_rec['array_path'];unit=np.load(noise_path,allow_pickle=False)['waveform'];noise64=rec['noise_gain']*unit;condition_noise=noise64.astype(np.float32);noisy=(arrays['clean']+noise64).astype(np.float32)
assert ah(noisy)==rec['model_input_float32_sha256']==proof['noisy_float32_sha256'];assert ah(condition_noise)==proof['noise_only_float32_sha256'];assert ah(arrays['clean'])==proof['clean_float64_sha256'];assert np.count_nonzero(arrays['clean'][:start])==np.count_nonzero(arrays['clean'][end:])==0
outside=np.ones(32000,bool);outside[start:end]=False;assert np.array_equal(noisy[outside],condition_noise[outside]);condition_proof={'conditions_exactly_equal_outside_known_clean_support':True,'outside_samples':int(outside.sum()),'outside_max_abs_difference':float(np.max(np.abs(noisy[outside]-condition_noise[outside]))),'clean_pre_and_post_padding_exact_zero':True,'noisy_input_sha256':ah(noisy),'noise_only_input_sha256':ah(condition_noise),'same_normalized_physical_noise_hash':proof['normalized_physical_noise_sha256'],'active_change_rms':float(np.sqrt(np.mean((noisy[start:end].astype(np.float64)-condition_noise[start:end])**2)))}
keys=['anchor_raw_sha256','normalized_anchor_sha256','refscale','seed','cpu_rng_sha256','cuda_rng_sha256','initial_xT_sha256','normalized_physical_noise_sha256'];geometry={k:native[k]==ablate[k] for k in keys};assert all(geometry.values()) and native['nfe']==ablate['nfe']==native['trace_steps']==ablate['trace_steps']==30
cachepaths={'native_off':F/native['ctc']['array_path'],'ablated_off':A/ablate['ctc']['array_path'],'y_pre':F/C['pre_score']['array_path']};cache_scores={'native_off':native['ctc'],'ablated_off':ablate['ctc'],'y_pre':C['pre_score']}
source_paths=[A/'metrics.json',A/'selection_frozen.json',A/'outputs/source_proofs.json',A/'config.yaml',F/'outputs/recipe_frozen.json',F/'outputs/audio_manifest.json',F/'outputs/per_sample.json',noise_path,vocabpath,modelcfg,Path(row['wav_path'])]+list(wavepaths.values())+list(cachepaths.values())
for base,r in [(F,C['frontend_audio']),(F,C['off_audio']),(A,native),(A,ablate)]:
 assert sha(base/r['raw_path'])==r['raw_sha256'] and sha(base/r['wav_path'])==r['wav_sha256'];source_paths.append(base/r['wav_path'])
source_paths=list(dict.fromkeys(source_paths));locked={str(p):sha(p) for p in source_paths};sources=[]
for arm,p in wavepaths.items():
 x=arrays[arm];assert x.shape==(32000,) and np.isfinite(x).all();sources.append({'arm':arm,'path':str(p),'file_sha256':sha(p),'array_sha256':ah(x),'dtype':str(x.dtype),'shape':list(x.shape),'sample_rate':SR,'alignment':'Common frozen carrier indices, no shift/resampling applied'})
# Rectangular one-sided Parseval: weighted |FFT|²/N² equals mean square.
def energy(x):
 x=np.asarray(x,np.float64);n=len(x);X=np.fft.rfft(x);freq=np.fft.rfftfreq(n,1/SR);weights=np.full(len(X),2.);weights[0]=1
 if n%2==0:weights[-1]=1
 spec=weights*np.abs(X)**2/n**2;ms=float(np.mean(x*x));assert abs(float(spec.sum())-ms)<=1e-12*max(ms,1e-20)+1e-25
 return {'samples':n,'rms':float(np.sqrt(ms)),'mean_square':ms,'highband_mean_square':float(spec[(freq>=4000)&(freq<=7500)].sum()),'parseval_absolute_error':abs(float(spec.sum())-ms)}
sine=np.sin(2*np.pi*5000*np.arange(SR)/SR);assert abs(energy(sine)['highband_mean_square']-.5)<1e-12
metrics={name:{arm:energy(x[a:b]) for arm,x in arrays.items()} for name,(a,b) in intervals.items()}
diffs={name:{pair:energy(arrays[left][a:b].astype(np.float64)-arrays[right][a:b]) for pair,left,right in [('ablated_minus_native','ablated_off','native_off'),('native_minus_y_pre','native_off','y_pre'),('ablated_minus_y_pre','ablated_off','y_pre')]} for name,(a,b) in intervals.items()}
frames=[]
for i,a in enumerate(range(0,32000,320)):
 for arm,x in arrays.items():frames.append({'frame':i,'arm':arm,'start_sample':a,'end_sample_exclusive':a+320,'start_s':a/SR,'end_s':(a+320)/SR,**energy(x[a:a+320])})
with (H/'frames20ms.csv').open('w',newline='') as f:
 w=csv.DictWriter(f,fieldnames=list(frames[0]));w.writeheader();w.writerows(frames)
# Recover exact convolution sample geometry from cached model config.
jump=1;rf=1
for k,s in zip(mconfig['conv_kernel'],mconfig['conv_stride']):rf+=(k-1)*jump;jump*=s
id2={v:k for k,v in vocab.items()};ctc={};s_id=vocab['s'];blank=vocab['<pad>']
for arm,p in cachepaths.items():
 a=np.load(p,allow_pickle=False);assert sha(p)==cache_scores[arm]['array_sha256'];logits=a['logits'];probs=a['probabilities'];assert logits.shape==probs.shape and logits.shape[1]==len(vocab);exp=np.exp(logits.astype(np.float64)-np.max(logits,axis=1,keepdims=True));assert np.max(np.abs(exp/exp.sum(axis=1,keepdims=True)-probs))<1e-6
 labels=a['candidate_labels'].tolist();assert dict(zip(labels,map(float,a['candidate_loglikelihoods'])))==cache_scores[arm]['all24_loglikelihoods'];arg=logits.argmax(-1);runs=[];i=0
 while i<len(arg):
  j=i+1
  while j<len(arg) and arg[j]==arg[i]:j+=1
  if arg[i]!=blank:runs.append({'token':id2[int(arg[i])],'token_id':int(arg[i]),'first_frame':i,'end_frame_exclusive':j,'first_conv_support_s':[i*jump/SR,(i*jump+rf)/SR],'last_conv_support_s':[(j-1)*jump/SR,((j-1)*jump+rf)/SR],'mean_token_posterior':float(probs[i:j,arg[i]].mean())})
  i=j
 centers=(np.arange(len(arg))*jump+rf/2)/SR;descriptive={}
 for name,(a0,b0) in intervals.items():
  ix=(centers>=a0/SR)&(centers<b0/SR);v=probs[ix,s_id];descriptive[name]={'selection':'fixed convolution receptive-field center falls in predeclared interval; transformer context is global','frame_indices':np.flatnonzero(ix).tolist(),'mean_s_posterior':float(v.mean()),'max_s_posterior':float(v.max()),'s_argmax_frames':np.flatnonzero(ix&(arg==s_id)).tolist()}
 ctc[arm]={'path':str(p),'shape':list(logits.shape),'blank_id':blank,'s_token_id':s_id,'greedy_collapsed_full_tokens':[r['token'] for r in runs],'greedy_nonblank_runs':runs,'center24_top1':cache_scores[arm]['top1_prediction'],'s_posterior':descriptive}
# Fixed post-support frame statistics include every eligible frame; no onset/window selection.
assert all(sha(p)==h for p,h in locked.items());after=subprocess.check_output(['git','status','--porcelain'],cwd=R).decode();(H/'git_status_after.txt').write_text(after)
out={'status':'completed','case_id':CID,'source_support_samples':[start,end],'source_support_seconds':[start/SR,end/SR],'declared_resampled_source_length':row['resampled_samples'],'sources':sources,'source_locks_before_after':locked,'all_sources_unchanged':True,'condition_proof':condition_proof,'native_ablated_geometry_match':geometry,'nfe_and_trace_steps':[30,30],'intervals':intervals,'waveform_metrics':metrics,'waveform_difference_metrics':diffs,'ctc_frame_geometry':{'conv_stride_samples':jump,'conv_receptive_field_samples':rf,'first_conv_support_seconds':[0,rf/SR],'center_formula':'(i*stride+receptive_field/2)/16000','limits':'Feature convolution support is not independent transformer or phoneme support. No user /s/ onset supplied.'},'ctc_cached':ctc,'full_clean_ctc_available':'not sought: only native/ablate and frontend cached logits required; clean waveform available','parseval_sine_calibration':True,'model_calls':0,'training_calls':0,'audio_outputs':0,'git_status_unchanged':git_before==after,'wall_seconds':time.monotonic()-T}
assert out['wall_seconds']<120;dump('metrics.json',out)
print(json.dumps({'support':out['source_support_seconds'],'intervals':metrics,'ctc':{k:{'tokens':v['greedy_collapsed_full_tokens'],'s':v['s_posterior']} for k,v in ctc.items()},'wall_seconds':out['wall_seconds']},indent=2,ensure_ascii=False))
