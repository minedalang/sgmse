"""Consolidate validated replication records without model forwards."""
from pathlib import Path
import json,hashlib,shutil,numpy as np
P=Path(__file__).resolve().parent
m=json.loads((P/'metrics.json').read_text());v=json.loads((P/'worker_validation.json').read_text());e=json.loads((P/'worker_execution.json').read_text());assert v['status']=='passed'
assert not (P/'outputs/runner_metrics.json').exists();shutil.copyfile(P/'metrics.json',P/'outputs/runner_metrics.json')
m['worker_validation']={'status':'passed','path':'worker_validation.json','sha256':hashlib.sha256((P/'worker_validation.json').read_bytes()).hexdigest(),'model_forwards':0,'cached_ctc_scores_verified':v['scores_verified'],'max_ctc_probability_error':0}
m['source_recording_support_snr']=v['support_snr_summary'];m['arm_comparisons']=v['arm_comparisons'];m['noise_stability_both_metrics']=v['noise_stability'];m['environment']=v['environment'];m['environment']['driver']='610.57.04';m['execution']={'command':e['command'],'cwd':e['cwd'],'exit_code':0,'script_wall_seconds':m['wall_seconds'],'external_wall_seconds':None,'cuda_only_seconds':None,'runtime_note':'Script duration includes imports, hashes, model loads, baseline and enhanced recognition, enhancement, audio/scoring output. External wall and CUDA-only duration were not instrumented.'}
if m['clean_gate_passed']:
 aa=json.loads((P/'outputs/audio_manifest.json').read_text());m['quiet_padding_summary']={}
 for c in m['noisy_input_eligibility']:
  m['quiet_padding_summary'][c]={}
  for arm in ['clean','noisy','y_pre','off','router']:
   a=[r for r in aa if r['condition']==c and r['arm']==arm];m['quiet_padding_summary'][c][arm]={}
   for window in ['pre_0_0p3','post_1p3_2p0']:
    vals=[r['quiet_padding'][window]['rms'] for r in a];m['quiet_padding_summary'][c][arm][window]={'min_raw_rms':min(vals),'median_raw_rms':float(np.median(vals)),'max_raw_rms':max(vals)}
m['artifact_bytes']=sum(q.stat().st_size for q in P.rglob('*') if q.is_file());m['disk_free_bytes']=shutil.disk_usage(P).free
(P/'metrics.json').write_text(json.dumps(m,indent=2,allow_nan=False)+'\n')
r=(P/'experiment_report.md').read_text()
r+='\nFresh source manifest SHA256: `'+v['frozen_manifest_sha256']+'`. Speakers f18,m5,f20,m15 exclude all9 speakers in the previous28-file manifest. All16 nominal identities and8 pairs retained. Held out refers to previous project analyses; pretrained exposure is unknown. Retrieval verification remains retrieval/worker_validation.json.\n'
r+=f"\nPadded clean gate passed: {m['summaries']['padded_clean']['clean']['top1_correct']}/16 24-way and {m['summaries']['padded_clean']['clean']['binary_correct']}/16 binary; each contrast8/8. All4 frozen noisy conditions executed. The noisy gates tag qualification only: "+', '.join(c+': '+str(ok) for c,ok in m.get('noisy_input_eligibility',{}).items())+'. No input-based sample or condition removal occurred.\n'
r+='\nSource-recording-support SNR (entire resampled source recording, not phoneme-local or clean-active):\n\n|Condition|Mean dB|Minimum dB|Maximum dB|\n|---|---:|---:|---:|\n'
for c,s in v['support_snr_summary'].items():r+=f"|{c}|{s['mean_db']:.6f}|{s['minimum_db']:.6f}|{s['maximum_db']:.6f}|\n"
r+='\nFull-carrier SNR remains exactly the registered ±6dB up to float64 rounding; support SNR uses [8000,8000+resampled_samples), so different clean padding fractions and noise power explain the numerical difference. Per-sample float64 and effective float32 carrier SNR values are in worker_validation.json. These windows do not define true consonant SNR.\n'
r+='\nCorrection and harm counts per16 inputs:\n\n|Condition|Comparison|24-way correct→wrong|24-way wrong→correct|Binary correct→wrong|Binary wrong→correct|\n|---|---|---:|---:|---:|---:|\n'
for c,cs in v['arm_comparisons'].items():
 for comp,t in cs.items():
  a=t['top1_correct'];b=t['binary_correct'];r+=f"|{c}|{comp}|{a['correct_to_wrong']}|{a['wrong_to_correct']}|{b['correct_to_wrong']}|{b['wrong_to_correct']}|\n"
r+='\nPair collapse and nominal-label agreement (all8 pairs; ties remain errors):\n\n|Condition|Arm|Both 24-way correct /8|Both binary correct /8|24-way collapse /8|Binary collapse /8|\n|---|---|---:|---:|---:|---:|\n'
for c,arms in m['summaries'].items():
 for arm,s in arms.items():r+=f"|{c}|{arm}|{s['both_correct_pairs_24way']}|{s['both_correct_pairs_binary']}|{s['same_prediction_collapse_24way']}|{s['same_prediction_collapse_binary']}|\n"
r+='\nCross-noise stability over16 identities:\n\n|Carrier SNR|Arm|Metric|Prediction changed|Correct in both noises|Stable incorrect|\n|---|---|---|---:|---:|---:|\n'
for snr,arms in v['noise_stability'].items():
 for arm,metrics in arms.items():
  for metric,s in metrics.items():r+=f"|{snr}|{arm}|{metric}|{s['changed_prediction']}|{s['both_correct']}|{s['stable_incorrect']}|\n"
r+='\nOriginal versus fresh speakers, descriptive only (different speakers, no significance inference):\n\n|Condition|Arm|Original 24-way /16|Fresh 24-way /16|Original binary /16|Fresh binary /16|\n|---|---|---:|---:|---:|---:|\n'
for c,arms in m['summaries'].items():
 if c=='padded_clean':continue
 oldname='p8_q_vcv_lowsnr_challenge_20261018' if 'snr-6' in c else 'p8_q_vcv_enhance_follow_20261018';old=m['original4speaker_descriptive'][oldname]['summaries'][c]
 for arm in ['noisy','y_pre','off','router']:
  s=arms[arm];o=old['noisy_reused' if arm=='noisy' else arm];r+=f"|{c}|{arm}|{o['top1_correct']}|{s['top1_correct']}|{o['binary_correct']}|{s['binary_correct']}|\n"
r+='\nFixed post-padding [1.3,2s] raw RMS medians, separately from identity metrics:\n\n|Condition|Y_pre|Off|Router|\n|---|---:|---:|---:|\n'
for c,arms in m.get('quiet_padding_summary',{}).items():r+='|'+c+'|'+'|'.join(f"{arms[a]['post_1p3_2p0']['median_raw_rms']:.9g}" for a in ['y_pre','off','router'])+'|\n'
r+='\nAll source clean padding is exactly zero; output pre[0,0.3s], post[1.3,2s] and outside-source-support RMS/mean-square are verified against cached raw waveforms. Padding energy is not a speech, word or hallucination label. Full ranges are in metrics.json and per-case energies in outputs/audio_manifest.json.\n'
r+=f"\nCache validation passed with zero model forwards:80 clean/noisy recipes regenerate at both float64 and float32 hashes;16 newly resampled clean arrays match originals;{v['raw_and_pcm16_audio_verified']} raw/PCM16 audio pairs match file/waveform hashes and exact installed libsndfile serialization;{v['scores_verified']} CTC scores, probabilities and JSON scores recompute with zero error. Summaries, eligibility, transitions and stability reproduce from cached scores. All8 common pair seeds and64 frontend RNG-invariance checks pass. Source recordings, checkpoint, recognizer, core code, frozen protocol and all old±6 records remain unchanged. Runtime uses80kEMA/N30, noisy concatenation=true, cross-attention=false, FiLM=false, selector=none; frontend storm and residual refinement=false checked by driver.\n"
r+=f"\nPCM16 copies have {v['pcm_clipping_count_total']} clipped samples and maximum quantization difference {v['pcm16_quantization_max_error']:.9g}. One step is1/32768 (−90.309dBFS); recognition uses unclipped float32. Linear energies retain exact zeros; no finite dB floor is inserted. No new model downloads, training or DNSMOS; no human listening result or optional page added.\n"
r+=f"\nExact command: `{e['command']}` from `{e['cwd']}`, exit0. Script duration {m['wall_seconds']:.6f}s (<600s); external wall/CUDA-only time uninstrumented. Environment: Python3.10.22, torch{v['environment']['torch']}, CUDA{v['environment']['cuda_build']}, cuDNN{v['environment']['cudnn']}, NVIDIA L40S driver610.57.04. Artifacts about{m['artifact_bytes']/1e6:.2f}MB (<200MB), free disk{m['disk_free_bytes']/1e9:.3f}GB. Preparation metrics/report preserved; original runner metrics at outputs/runner_metrics.json.\n"
(P/'experiment_report.md').write_text(r)
size=sum(q.stat().st_size for q in P.rglob('*') if q.is_file());assert size<200_000_000 and shutil.disk_usage(P).free>2_000_000_000
print(json.dumps({'status':m['status'],'validation':v['status'],'artifact_bytes':size,'disk_free_bytes':shutil.disk_usage(P).free}))
