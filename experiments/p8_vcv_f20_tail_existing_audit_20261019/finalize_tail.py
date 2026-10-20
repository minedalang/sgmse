"""CPU cached-score extension, fixed-grid scientific figure and report."""
from pathlib import Path
import json,hashlib,csv,time
import numpy as np
H=Path(__file__).resolve().parent;R=H.parent.parent;A=R/'experiments/p8_vcv_speech_component_ablation_20261019';F=R/'experiments/p8_q_vcv_speaker_replication_20261019';T=time.monotonic()
def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
D=read(H/'metrics.json');feedback=A/'listening/manual_listener_feedback_completion_20261019.json';human=next(r for r in read(feedback)['responses'] if r['ordinal']==5);D['human_feedback']={'path':str(feedback),'sha256':sha(feedback),'response':human,'timing':'User reports末尾 only, exact sound onset unknown'};D['source_locks_before_after'][str(feedback)]=sha(feedback)
vp=next(Path(p) for p in D['source_locks_before_after'] if p.endswith('/vocab.json'));vocab=read(vp);id2={i:t for t,i in vocab.items()};s_id=vocab['s'];blank=vocab['<pad>'];stride=D['ctc_frame_geometry']['conv_stride_samples'];rf=D['ctc_frame_geometry']['conv_receptive_field_samples']
# Clean full-vocabulary cached CTC does exist; extend without a forward call.
cs=next(s for s in read(F/'outputs/per_sample.json') if s['sample_id']=='f20aka1' and s['condition']=='padded_clean' and s['arm']=='baseline');cp=F/cs['array_path'];assert sha(cp)==cs['array_sha256'];D['source_locks_before_after'][str(cp)]=sha(cp);ca=np.load(cp,allow_pickle=False);arg=ca['logits'].argmax(-1);probs=ca['probabilities'];runs=[];i=0
while i<len(arg):
 j=i+1
 while j<len(arg) and arg[j]==arg[i]:j+=1
 if arg[i]!=blank:runs.append({'token':id2[int(arg[i])],'token_id':int(arg[i]),'first_frame':i,'end_frame_exclusive':j,'first_conv_support_s':[i*stride/16000,(i*stride+rf)/16000],'last_conv_support_s':[(j-1)*stride/16000,((j-1)*stride+rf)/16000],'mean_token_posterior':float(probs[i:j,arg[i]].mean())})
 i=j
centers=(np.arange(len(arg))*stride+rf/2)/16000;sp={}
for name,(a,b) in D['intervals'].items():
 ix=(centers>=a/16000)&(centers<b/16000);sp[name]={'frame_indices':np.flatnonzero(ix).tolist(),'mean_s_posterior':float(probs[ix,s_id].mean()),'max_s_posterior':float(probs[ix,s_id].max()),'s_argmax_frames':np.flatnonzero(ix&(arg==s_id)).tolist()}
D['ctc_cached']['clean']={'path':str(cp),'shape':list(ca['logits'].shape),'center24_top1':cs['top1_prediction'],'greedy_collapsed_full_tokens':[r['token'] for r in runs],'greedy_nonblank_runs':runs,'s_posterior':sp};D['full_clean_ctc_available']=True
ctcrows=[]
for arm,c in D['ctc_cached'].items():
 a=np.load(c['path'],allow_pickle=False);p=a['probabilities'];assert p.shape[0]==99
 for i in range(len(p)):ctcrows.append({'arm':arm,'frame':i,'conv_start_s':i*stride/16000,'conv_end_s':(i*stride+rf)/16000,'conv_center_s':(i*stride+rf/2)/16000,'greedy_token':id2[int(a['logits'][i].argmax())],'s_posterior':float(p[i,s_id]),'blank_posterior':float(p[i,blank])})
with (H/'cached_ctc_frames.csv').open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(ctcrows[0]));w.writeheader();w.writerows(ctcrows)
# No data-derived interval or threshold selection: all100 disjoint20ms frames.
plotstatus='unavailable'
try:
 import matplotlib
 matplotlib.use('Agg')
 import matplotlib.pyplot as plt
 rows=list(csv.DictReader((H/'frames20ms.csv').open()));fig,axs=plt.subplots(2,1,figsize=(9,5.3),sharex=True,layout='constrained')
 for arm,color in [('clean','#333333'),('y_pre','#247ba0'),('native_off','#e38b2c'),('ablated_off','#b5356a')]:
  rr=[r for r in rows if r['arm']==arm];tt=[(float(r['start_s'])+float(r['end_s']))/2 for r in rr]
  for ax,key in zip(axs,['rms','highband_mean_square']):ax.plot(tt,[float(r[key]) for r in rr],label=arm,color=color,lw=1.2)
 for ax in axs:
  ax.set_yscale('symlog',linthresh=1e-7 if ax is axs[0] else 1e-13);ax.axvline(D['source_support_seconds'][1],c='black',ls='--',lw=.9,label='declared source end');ax.axvline(D['source_support_seconds'][1]+.1,c='grey',ls=':',lw=1,label='source end +100 ms');ax.grid(alpha=.2);ax.set_ylim(bottom=0)
 axs[0].set_ylabel('20 ms frame RMS');axs[1].set_ylabel('4–7.5 kHz mean square');axs[1].set_xlabel('Time in frozen 2 s carrier (s)');axs[0].legend(fontsize=7,ncol=3);axs[0].set_title('Existing f20aka1 / N2 −6 dB: fixed 20 ms grid; user tail onset unspecified');fig.savefig(H/'tail_fixed_grid.png',dpi=140);plt.close(fig);plotstatus='saved'
except ImportError as e:plotstatus='unavailable: '+str(e)
D['scientific_plot_status']=plotstatus
text='''# Case5 existing tail audit

Completed CPU-only on saved artifacts, with zero model initialization, inference/training, downloads or audio output. User's case5 comment is recorded in the original completion JSON and hashed here:

> '''+human['comment_verbatim']+'''

The listener supplied no exact /s/ timestamp. We use only the frozen clean recording support and a prespecified100ms guard, never a window selected to match the report. Case4's distinct t→k observation is not analyzed as another audio case here.

The actual recorded clean length is6881 resampled samples, inserted at8000 in a32000-sample16kHz carrier. Declared source interval is[8000,14881),0.5–0.9300625s. Clean padding before and after this interval is bitwise zero. Raw clean(float64),Y_pre/native/ablated(float32),original manifests,noise recipe,source WAV,source proof,vocabulary,model JSON and cached logits are real and hashed before/after. Native replay raw/WAV hashes exactly match the original fresh-speaker OFF. All arrays share carrier indexing; no resampling or delay compensation was applied. This proves preparation/crop provenance, not absence of generated timing shifts.

|Fixed interval|Arm|RMS|4–7.5kHz mean-square|
|---|---|---:|---:|
'''
for name,m in D['waveform_metrics'].items():
 a,b=D['intervals'][name]
 for arm,v in m.items():text+=f"|{a/16000:.7f}–{b/16000:.1f}s|{arm}|{v['rms']:.8g}|{v['highband_mean_square']:.8g}|\n"
text+='\n|Fixed interval|Ablated−native difference RMS|Ablated/native highband ratio|\n|---|---:|---:|\n'
for name,m in D['waveform_metrics'].items():text+=f"|{name}|{D['waveform_difference_metrics'][name]['ablated_minus_native']['rms']:.8g}|{m['ablated_off']['highband_mean_square']/m['native_off']['highband_mean_square']:.4f}|\n"
text+='''
The highband increase is measured in both fixed post-support intervals; it is not a phoneme identity detector. Existing coarse1.3–2s RMS decreases from8.16325e−5 to7.00078e−5 despite the listener's extra/s/report. Lower overall energy does not contradict additional structured sound. All100 disjoint20ms frames per arm, including zero-clean frames, are retained in frames20ms.csv. The figure shows that entire fixed grid without labelling an acoustic /s/ onset.

![Fixed20ms whole-carrier waveform statistics](tail_fixed_grid.png)

Highband uses an unwindowed rectangular one-sided FFT, inclusive4–7.5kHz bins, weight1 at DC/Nyquist and2 at other positive frequencies: sum(weight×|FFT|²)/N². Total weighted spectrum was checked against mean(x²) for every frame/interval and difference waveform. A5kHz unit sine gives mean-square0.5. This is finite-segment spectral energy with possible leakage, not a perceptual sibilance score.

|Existing cache|24-way center class|Full greedy collapsed nonblank tokens|Guarded-tail mean /s/ posterior|Guarded-tail max /s/ posterior|
|---|---|---|---:|---:|
'''
for arm,c in D['ctc_cached'].items():
 z=c['s_posterior']['after_declared_support_plus100ms'];text+=f"|{arm}|{c['center24_top1']}|{' '.join(c['greedy_collapsed_full_tokens'])}|{z['mean_s_posterior']:.8g}|{z['max_s_posterior']:.8g}|\n"
text+='''
Both native and ablated center scores remain ch, yet their cached full-vocabulary greedy sequences differ at the tail. In ablated output,s is argmax only at frame53 with posterior0.67217; native and frontend have no s-argmax tail frames. Model convolution geometry is stride320samples/receptive field400samples: frame53 has support1.060–1.085s, center1.0725s. This is a cached-model proxy location, not a listener-supplied boundary; subsequent transformer context is global. Full99-frame vocabulary logits/probabilities exist for native/ablated/frontend/clean, and all per-frame /s/,blank and greedy tokens are saved in cached_ctc_frames.csv. No extra classifier calls occurred. The same recognizer and its free decode are not an independent human measurement, and its literal full-token string need not be phonetically accurate. Center-only three-token candidate scoring misses unconstrained extra structure; unchanged ch does not establish full-content equality.

Reconstructing the frozen float64 clean+physical noise and casting tofloat32 reproduces the original noisy hash; casting the same physical noise reproduces the actual ablated-condition hash. The two conditions are exactly identical at every sample outside[8000,14881),including both fixed tail intervals. Native/ablate raw and normalized anchor,reference,seed,CPU/CUDA RNG,initial xT and normalized physical-noise hashes all match; NFE/trace counts are30/30. Thus changed output padding can be a nonlocal response to removing earlier speech while preserving local tail condition and anchor. This static provenance fact does not identify the phoneme-generation mechanism.

The intervention removes the whole VCV only from the direct noisy condition, while speech remains in the frontend anchor. It is clean-aware OOD, not a deployable method, isolated consonant removal or proof of independent noisy recoverability. The listener report and cached proxy are consistent about additional tail structure; this single selected,nonblind example provides no insertion rate or population improvement estimate. No new listening request/page/cases were created. Root Research Lead retains interpretation and next-direction decisions; root plan was not edited. Existing dirty files were preserved. Reproduce audit_tail.py then finalize_tail.py only in a distinct fresh destination; the first script refuses existing metrics. All source locks rechecked before final record.
'''
(H/'experiment_report.md').write_text(text)
assert all(sha(Path(p))==h for p,h in D['source_locks_before_after'].items());D['finalizer_cpu_wall_seconds']=time.monotonic()-T;D['all_sources_unchanged']=True
for _ in range(4):
 (H/'metrics.json').write_text(json.dumps(D,indent=2,ensure_ascii=False,allow_nan=False)+'\n');D['artifact_bytes']=sum(p.stat().st_size for p in H.rglob('*') if p.is_file())
assert D['artifact_bytes']<2000000 and D['wall_seconds']+D['finalizer_cpu_wall_seconds']<120
print(json.dumps({'artifact_bytes':D['artifact_bytes'],'cpu_wall_seconds':D['wall_seconds']+D['finalizer_cpu_wall_seconds'],'plot':plotstatus,'immutable_locks':len(D['source_locks_before_after']),'all_unchanged':True},indent=2))
