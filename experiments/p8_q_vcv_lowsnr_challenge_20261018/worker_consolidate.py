"""Consolidate validated caches and source-support SNR; no model execution."""
from pathlib import Path
import json,hashlib,shutil,numpy as np
P=Path(__file__).resolve().parent;PREV=P.parent/'p8_q_vcv_fixed_noise_20261018'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(p,v):p.write_text(json.dumps(v,indent=2,allow_nan=False)+'\n')
m=json.loads((P/'metrics.json').read_text());v=json.loads((P/'worker_validation.json').read_text());rr=json.loads((P/'outputs/per_sample.json').read_text());aa=json.loads((P/'outputs/audio_manifest.json').read_text());baseline=json.loads((PREV/'recognition_outputs/per_sample.json').read_text());b={(r['condition'],r['sample_id']):r for r in baseline}
assert m['status']=='completed' and v['status']=='passed'
assert not (P/'outputs/runner_metrics.json').exists();shutil.copyfile(P/'metrics.json',P/'outputs/runner_metrics.json')
f=json.loads((PREV/'inputs/recipe_frozen.json').read_text());support_audit=json.loads((PREV/'support_snr_audit.json').read_text());support_rows={(r[0],r[1]):r for r in support_audit['records']}
caches={r.get('sample_id',r.get('noise_id')):np.load(PREV/r['array_path'],allow_pickle=False)['waveform'] for r in f['clean']+f['noise']};sources={r['sample_id']:r for r in f['clean']}
ss=[]
for r in f['recipes']:
 if r['condition'] not in m['summaries']:continue
 c=caches[r['sample_id']];n=r['noise_gain']*caches[r['noise_id']];end=8000+sources[r['sample_id']]['resampled_samples'];s=float(10*np.log10(np.mean(np.square(c[8000:end],dtype=np.float64))/np.mean(np.square(n[8000:end],dtype=np.float64))))
 assert s==support_rows[(r['condition'],r['sample_id'])][9];ss.append(s)
support={'source_audit_path':str(PREV/'support_snr_audit.json'),'source_audit_sha256':sha(PREV/'support_snr_audit.json'),'n':32,'minimum_db':min(ss),'mean_db':float(np.mean(ss)),'maximum_db':max(ss),'definition':support_audit['support_definition'],'interpretation':'Full two-second carrier SNR=-6dB. Source-recording-support SNR excludes padded clean zeros and measures the entire recording, not true consonant or clean-active SNR. Different windows explain the numerical difference; frozen recipe is unchanged.'}
dump(P/'support_snr_validation.json',support)
for c,arms in m['transitions_from_noisy'].items():
 for arm,tr in arms.items():
  rs=[r for r in rr if r['condition']==c and r['arm']==arm]
  for key in ['top1_correct','binary_correct']:
   expected={'correct_to_wrong':sum(b[(c,r['sample_id'])][key] and not r[key] for r in rs),'wrong_to_correct':sum(not b[(c,r['sample_id'])][key] and r[key] for r in rs),'correct_to_correct':sum(b[(c,r['sample_id'])][key] and r[key] for r in rs),'wrong_to_wrong':sum(not b[(c,r['sample_id'])][key] and not r[key] for r in rs)}
   assert tr[key]==expected
m.update(worker_validation={'status':'passed','path':'worker_validation.json','sha256':sha(P/'worker_validation.json'),'noisy_transitions_verified':True,'all_model_forwards':0},arm_comparisons=v['arm_comparisons'],noise_identity_stability=v['noise_stability'],source_recording_support_snr=support,environment=v['environment'])
m['environment']['driver']='610.57.04'
m['execution']={'command':'PATH="/data/coding/sgmse/env/bin:$PATH" /data/coding/sgmse/env/bin/python experiments/p8_q_vcv_lowsnr_challenge_20261018/run_challenge.py','cwd':'/data/coding/sgmse','exit_code':0,'script_wall_seconds':m['wall_seconds'],'external_process_wall_seconds':None,'cuda_only_seconds':None,'runtime_note':'Script time includes imports, hashing, loads, enhancement, recognition and output writing. External wall/CUDA-only time was not instrumented.'}
m['quiet_padding_summary']={}
for c in m['summaries']:
 m['quiet_padding_summary'][c]={}
 for arm in ['clean','noisy','y_pre','off','router']:
  ar=[a for a in aa if a['condition']==c and a['arm']==arm];m['quiet_padding_summary'][c][arm]={}
  for window in ['pre_0_0p3','post_1p3_2p0']:
   x=[a['quiet_padding'][window]['rms'] for a in ar]
   m['quiet_padding_summary'][c][arm][window]={'minimum_raw_rms':min(x),'median_raw_rms':float(np.median(x)),'maximum_raw_rms':max(x)}
m['output_bytes']=sum(q.stat().st_size for q in P.rglob('*') if q.is_file());m['disk_free_bytes']=shutil.disk_usage(P).free
dump(P/'metrics.json',m)
report=(P/'experiment_report.md').read_text()
report+=f"\nFull-carrier SNR remains −6dB over the complete2s input. Independently verified SNR over source recording support has mean {support['mean_db']:.6f}dB, range [{support['minimum_db']:.6f}, {support['maximum_db']:.6f}]dB. The support is [8000,8000+resampled_samples), containing the entire VCV recording; it is not phoneme-local or clean-active SNR. Clean energy is concentrated in this shorter support, explaining the difference without any recipe change. See support_snr_validation.json and the immutable prior support_snr_audit.json. Both original −6dB input gates remain failed; this separate known-label challenge does not requalify them.\n"
report+='\nTransitions from noisy (per16 inputs):\n\n|Condition|Arm|24-way correct→wrong|24-way wrong→correct|Binary correct→wrong|Binary wrong→correct|\n|---|---|---:|---:|---:|---:|\n'
for c,arms in m['transitions_from_noisy'].items():
 for arm,tr in arms.items():
  t=tr['top1_correct'];q=tr['binary_correct'];report+=f"|{c}|{arm}|{t['correct_to_wrong']}|{t['wrong_to_correct']}|{q['correct_to_wrong']}|{q['wrong_to_correct']}|\n"
report+='\nBackend transitions (per16 inputs):\n\n|Condition|Comparison|24-way correct→wrong|24-way wrong→correct|Binary correct→wrong|Binary wrong→correct|\n|---|---|---:|---:|---:|---:|\n'
for c,comps in v['arm_comparisons'].items():
 for comp,tr in comps.items():
  t=tr['top1_correct'];q=tr['binary_correct'];report+=f"|{c}|{comp}|{t['correct_to_wrong']}|{t['wrong_to_correct']}|{q['correct_to_wrong']}|{q['wrong_to_correct']}|\n"
report+='\nPair collapse counts:\n\n|Condition|Arm|Same 24-way prediction /8|Same binary prediction /8|\n|---|---|---:|---:|\n'
for c,arms in m['summaries'].items():
 for arm,s in arms.items():report+=f"|{c}|{arm}|{s['same_prediction_collapse_24way']}|{s['same_prediction_collapse_binary']}|\n"
report+='\nCross-noise stability over16 identical nominal identities:\n\n|Arm|Metric|Same prediction|Correct in both noises|Stable incorrect|Changed prediction|\n|---|---|---:|---:|---:|---:|\n'
for arm,vals in v['noise_stability'].items():
 for metric,r in vals.items():report+=f"|{arm}|{metric}|{r['same_prediction']}|{r['both_correct']}|{r['stable_incorrect']}|{16-r['same_prediction']}|\n"
report+='\nFixed post-padding [1.3,2.0s] raw RMS, separately from identity scores:\n\n|Condition|Arm|Minimum|Median|Maximum|\n|---|---|---:|---:|---:|\n'
for c,arms in m['quiet_padding_summary'].items():
 for arm,windows in arms.items():
  q=windows['post_1p3_2p0'];report+=f"|{c}|{arm}|{q['minimum_raw_rms']:.9g}|{q['median_raw_rms']:.9g}|{q['maximum_raw_rms']:.9g}|\n"
report+='\nRaw pre-padding and full outside-source-support energy remain in outputs/audio_manifest.json; fixed pre[0,0.3s] clean padding and post padding contain exact zeros. Padding energy measurements do not label generated speech, words or hallucinations.\n'
report+=f"\nValidation passed without model forwards: all32 recipe inputs regenerate exactly; all160 finite32000-sample float32 reference/output arrays and their PCM16 WAV serializations match; all2304 enhanced CTC scores and1152 reused baseline CTC scores/probabilities reproduce exactly. Summaries, binary-only cases and noisy transition counts reproduce from cached scores. All frozen source/checkpoint/config/runner/core-code and old+6 artifact hashes remain unchanged. All8 pair seeds are shared across identities/noises/arms; all32 frontends preserve CPU/CUDA RNG state. Runtime flags retain noisy concatenation=true, cross-attention=false, FiLM=false, selector=none, EMA80k/N30.\n"
report+=f"\nPrimary recognition uses unclipped raw float32. Listening PCM16 clips {v['pcm_clipping_count_total']} samples and differs from clipped raw by at most {v['pcm16_quantization_max_error']:.9g}. Conversion is checked using installed libsndfile byte serialization. One quantization step is1/32768 (−90.309dBFS); linear padding RMS/mean-square are retained and exact zero has−infinity dBFS rather than an arbitrary finite floor. No human listening result was collected.\n"
report+=f"\nExact command: `{m['execution']['command']}` from `/data/coding/sgmse`, exit0. Script time {m['wall_seconds']:.6f}s (<600s); external process wall/CUDA-only time was not instrumented. Python3.10.22, torch{v['environment']['torch']}, CUDA{v['environment']['cuda_build']}, cuDNN{v['environment']['cudnn']}, L40S, driver610.57.04. No training/download/DNSMOS. Artifact size about{m['output_bytes']/1e6:.2f}MB (<100MB); free disk{m['disk_free_bytes']/1e9:.3f}GB. Preparation records preserved; original runner metrics at outputs/runner_metrics.json. Optional listening.html is available, with no requested or assumed human judgment.\n"
(P/'experiment_report.md').write_text(report)
size=sum(q.stat().st_size for q in P.rglob('*') if q.is_file());assert size<100_000_000 and shutil.disk_usage(P).free>2_000_000_000
print(json.dumps({'status':'completed_validated','support_snr':support,'artifact_bytes':size,'disk_free_bytes':shutil.disk_usage(P).free},indent=2))
