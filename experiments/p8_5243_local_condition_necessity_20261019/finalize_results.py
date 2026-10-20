"""CPU-only preregistered comparison report and compact scientific figure."""
from pathlib import Path
import json,time
import numpy as np
H=Path(__file__).resolve().parent;D=json.loads((H/'metrics.json').read_text());V=json.loads((H/'artifact_validation.json').read_text());S=json.loads((H/'selection_frozen.json').read_text());T=time.monotonic()
assert D['status']=='completed' and V['status']=='passed'
names=['native','target','left','right'];regions={'target':(309600,312800),'left':(300960,304160),'right':(317280,320480)};raw=np.load(H/'outputs/native_raw.npy');rms={n:float(np.sqrt(np.mean(raw[a:b].astype(np.float64)**2))) for n,(a,b) in regions.items()};relative={n:{r:D['variants'][n]['output_delta_vs_native'][r]['rms']/rms[r] for r in regions} for n in names}
p0=D['variants']['native']['fixed_d_metrics']['fixed_posterior_mean'];near0=D['variants']['native']['fixed_d_metrics']['neighborhood_posterior_mean'];delta={n:p0-D['variants'][n]['fixed_d_metrics']['fixed_posterior_mean'] for n in names};margin=delta['target']-max(delta['left'],delta['right']);near_delta={n:near0-D['variants'][n]['fixed_d_metrics']['neighborhood_posterior_mean'] for n in names}
plot='not available'
try:
 import matplotlib
 matplotlib.use('Agg')
 import matplotlib.pyplot as plt
 fig,ax=plt.subplots(1,2,figsize=(10,3.6),layout='constrained');colors={'native':'#333333','target':'#b5356a','left':'#247ba0','right':'#e38b2c'}
 for n in names:
  p=np.load(H/'outputs'/(n+'_ctc_probabilities.npy'));ax[0].plot(range(25,46),p[25:46,12],marker='.',lw=1,label=n,color=colors[n])
 ax[0].axvline(35,color='gray',ls='--',lw=.8);ax[0].set(xlabel='Frozen CTC frame index',ylabel='Cached P(d)',title='Fixed neighborhood [25,46); no smoothing');ax[0].set_ylim(bottom=0);ax[0].legend(fontsize=8);ax[0].grid(alpha=.2)
 xx=np.arange(3)
 for i,n in enumerate(['target','left','right']):ax[1].bar(xx+(i-1)*.25,[relative[n][r] for r in regions],width=.25,label=n,color=colors[n])
 ax[1].set_xticks(xx,labels=['target region','left region','right region']);ax[1].set_yscale('log');ax[1].set(ylabel='RMS(output−native) / native region RMS',title='All three preregistered regions');ax[1].legend(fontsize=8);ax[1].grid(axis='y',alpha=.2);fig.suptitle('Natural5243 local condition edits: selected OOD diagnostic, not phoneme correctness',fontsize=10);fig.savefig(H/'fixed_preregistered_comparison.png',dpi=140);plt.close(fig);plot='saved'
except ImportError as e:plot='unavailable: '+str(e)
text=f'''# Natural5243 local-condition diagnostic

Completed in {D['supervisor_seconds']:.5f}seconds,0frontend/4OFF N30/4short existing-recognizer calls. No failures,inference retries,Router,training,downloads,window/threshold/seed changes or additional cases. All four results are retained. Artifact size is about13.8MB,below30MB;runtime is below240seconds.

The preregistered target-necessity direction is not supported by this recognizer measurement: target removal increases fixed/nearby support,while the equal-energy left control reduces it. All four still identify d as top1 at frame35. This does not establish a true phoneme deletion or recovery outcome.

|Arm|FixedP(d),[35,36)|Nearbymax,[25,46)|Nearbymean|Peak frame|d-top1 nearby frames|Δ=native−arm,fixed|
|---|---:|---:|---:|---:|---|---:|
'''
for n in names:
 z=D['variants'][n]['fixed_d_metrics'];text+=f"|{n}|{z['fixed_posterior_mean']:.8f}|{z['neighborhood_posterior_max']:.8f}|{z['neighborhood_posterior_mean']:.8f}|{z['neighborhood_peak_frame']}|{z['neighborhood_d_top1_frames']}|{delta[n]:+.8f}|\n"
text+=f'''
Preregistered contrast Δtarget−max(Δleft,Δright)={margin:+.8f}. The left control has the greatest downward fixed posterior effect (+0.25511259);target has −0.10960472 and right −0.08953822. Nearbymean changes have the same direction:target {near_delta['target']:+.8f},left {near_delta['left']:+.8f},right {near_delta['right']:+.8f}. Fixed-frame and nearbymax peaks remain35,so there is no measured peak relocation within the frozen neighborhood. No success threshold,p-value,bootstrap,PER or recovery rate is assigned. This is a single selected event and cannot establish noise irrelevance or that the backend never uses direct noisy speech.

Full serialized-clip free greedy CTC sequences,not human transcripts:

'''
for n in names:text+=f"- {n}: `{D['variants'][n]['greedy_text']}`\n"
text+='''
The full sequences vary outside the fixed d target,so single-frame posterior is not a full-content score. Candidate interpretations must retain classifier/global normalization/context sensitivity; no new listening request/page was created.

![Frozen frame and region comparison](fixed_preregistered_comparison.png)

The left plot shows every frame25–45 with no smoothing or new window selection. The right plot normalizes output-difference RMS to native RMS in each corresponding fixed region;this visual scale is descriptive,not a statistical effect size.

|Edited condition|OutputΔRMS in target19.35–19.55s|OutputΔRMS in left18.81–19.01s|OutputΔRMS in right19.83–20.03s|Outside all three RMS|
|---|---:|---:|---:|---:|
'''
for n in names:
 z=D['variants'][n]['output_delta_vs_native'];text+=f"|{n}|{z['target']['rms']:.8g}|{z['left']['rms']:.8g}|{z['right']['rms']:.8g}|{z['outside_all_three_windows']['rms']:.8g}|\n"
text+='''
Target removal changes target waveform RMS by1.67644e−3,whereas left/right removal changes that same target region by5.63878e−6/7.99487e−6. Despite this much smaller target-region waveform change,the left control lowers targetposterior most. Outputs also have nonzero differences beyond edited windows. Waveform magnitude is not a phoneme correctness score,and this contrast does not isolate acoustic evidence extraction from recognizer context or clip-wide normalization.

|Arm|2sPCM clip mean|2sPCM clip RMS|2sPCM clip std|Processor input RMS|
|---|---:|---:|---:|---:|
'''
for n in names:
 z=D['variants'][n]['measurement_processing'];c=z['clip_pcm_float32'];text+=f"|{n}|{c['mean']:.8g}|{c['rms']:.8g}|{c['std']:.8g}|{z['processor_input']['rms']:.8g}|\n"
text+='''
All recognizer calls used exactly the original18.75–20.75s serializedPCM16 clip and same official Wav2Vec2Processor/ForCTC,with no added normalization. Processor performs clip-wide normalization;the encoder contextualizes frames over the clip. Therefore local audio edits can alter target posterior through global input statistics/context even where the waveform difference is tiny. No alternate normalization or re-alignment was run. Cached model geometry is99×392;all logits,full softmax probabilities and greedy IDs are retained,including non-target classes.

Input design was fixed before outputs:200ms target[309600,312800),left[300960,304160),right[317280,320480),internal320-sample(20ms)raised-cosine edges. Control selection used clean energy/time only;alpha target1,left0.8688384058650895,right0.9666585492603262. All ideal removals match squaredL2energy0.14454430003522467;actualfloat32relative errors are≤3.35e−9. Each condition is bitwise original outside its selected window and retains originalpeak0.431304931640625. Physical-noise amplitude/source remains fixed within recorded float32roundoff and PCM mixture bounds. Masks,ideal removed clean segments and actual rounded segment deltas are saved;full input conditions are reproducibly hashed instead of duplicating sourceaudio.

Native backend gate passed before any intervention:fullraw exactly matches recovered attempt02 baseline,full30sPCM bytes/samples and2scontext bytes/samples exactly match historicalOFF. Every variant uses the same frozenfloatanchor,reference,restored before_backend and before_sampler CPU/CUDA RNG,first predictor-before x_t hash/time,post-predictor callback RNG and end-backend RNG consumption. All4 haveNFE30/30trace callbacks and same sampler settings. Later reverse trajectory equality is neither required nor implied. Callback RNG is predictor-after and is not mislabeled initial-state RNG.

Native recognizer gate also passed:all oldcached target-token columns match with maximumdifference0.0 (tolerance1e−5),and all99frameargmax IDs agree. Thus current native measurement is exactly faithful to the prior cache for stored columns. New logits/full probabilities are retained for all4outputs. Independent CPU validation rehashes56nonweight sources,verifies actual runtimepre/post hashes for checkpoint and classifier weights,all12frozen execution files and56supervisor-manifest entries;it reconstructs3input conditions and all energy/outside/peak/noise guards,verifies raw/PCM serialization/native gates,paired anchor/RNG/firststate guards,and recomputes all4softmax arrays/fixed/nearby scores/greedy IDs/region waveform deltas. Maximum float64-versus-savedfloat32softmax difference is<4e−7. See artifact_validation.json and validate_completed_run.py. The initial validation draft's syntax-only mistake was detected by AST check before validation;it remains as a supervisor-captured artifact,with the corrected standalone checker used for the passed audit. No model/execution sources were modified.

Actual source/core/checkpoint/classifier58locks were hash-checked before and after execution,all unchanged. The12prepared execution files,selection andRootanalysis_plan remain frozen. Existing dirty workspace,data/weights/listeningserver were preserved. No rootplan edit,commit or push.

Interpretation boundary:the200ms target brackets an oldcleanCTCproxy,not manually isolated/d/;anchor retains speech whiledirectcondition loses knowncleancontribution,making the jointinputOOD. Two position controls match perturbation energy but not phonetic content or intelligibility. The observed targetincrease/leftdecrease does not support the fixed target-necessity direction and points to positional/contextual proxy dependence within this diagnostic;it does not prove a physical mechanism or natural noisy independent recoverability. All four calls finish the registered scope;no posthoc new seeds,boundaries,strengths or controls are introduced. Research Lead decides next direction separately.
'''
(H/'experiment_report.md').write_text(text)
summary={'status':'completed_validated','fixed_posterior_delta_native_minus_arm':delta,'nearby_mean_delta_native_minus_arm':near_delta,'preregistered_target_minus_max_control_fixed_delta':margin,'target_specific_decrease_direction_supported':False,'output_delta_RMS_relative_to_native_region_RMS':relative,'native_region_RMS':rms,'plot_status':plot,'finalizer_seconds':time.monotonic()-T,'model_calls_added':0,'supervisor_seconds':D['supervisor_seconds'],'calls':D['calls']}
for _ in range(4):
 (H/'validated_summary.json').write_text(json.dumps(summary,indent=2)+'\n');summary['final_artifact_bytes']=sum(p.stat().st_size for p in H.rglob('*') if p.is_file() and not p.is_symlink())
assert summary['final_artifact_bytes']<30000000
print(json.dumps(summary,indent=2))
