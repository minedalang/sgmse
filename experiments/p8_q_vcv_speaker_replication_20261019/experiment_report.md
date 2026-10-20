# Fresh-speaker VCV replication

## Research Lead decision: original Router gain did not replicate

The fresh four speakers do not reproduce the positive net Router identity
effect seen in the original four. Router introduces a new24-way g→dj error
for m5aga1 in N1−6dB, and a t→d error for m5ata1 in N1+6dB; N2 output
predictions are unchanged from off. Retain these negative results rather than
searching more seeds or speakers until a positive average appears.

Stop expanding the claim that Router reliably improves consonant identity
or training an auxiliary loss on these apparent proxy gains. This conclusion
is limited to the controlled corpus, two fixed noises and current recognizer;
it does not establish general content harm on natural DNS speech. Central
candidate identity, unwanted padding sound, human intelligibility and quality
remain distinct outcomes.

The next mechanism question concerns use of useful noisy evidence versus
noise-induced structure. A raw noisy swap across k/g examples would also
change global normalization: checked f6 and m5 pairs have different input
peaks. Such a swap does not hold the normalized frontend anchor or noise
gain fixed. Any later causal comparison must control those quantities and
first pass an exact unmodified-baseline equivalence check. No new training
or additional noisy condition selection follows automatically from this run.

Status: completed. Classifier forwards: 272.

Fresh speakers held out from previous project analyses only; pretrained corpus exposure is unknown. Full two-second carrier SNR is not phoneme-local SNR; quiet fractions differ from historical30second nominal training SNR. External nominal labels are not model truth. No human recoverability, ignored input evidence, natural DNS generalization, significance, confidence interval, WER or DNSMOS claim. Failed noisy gates remain unqualified but all frozen conditions execute if the clean diagnostic gate passes.

|Condition|Arm|24-way|Binary|Both correct binary pairs|
|---|---|---:|---:|---:|
|padded_clean|clean|16/16|16/16|8/8|
|N1_snr-6|noisy|8/16|10/16|2/8|
|N1_snr-6|y_pre|14/16|15/16|7/8|
|N1_snr-6|off|15/16|15/16|7/8|
|N1_snr-6|router|14/16|15/16|7/8|
|N2_snr-6|noisy|6/16|13/16|5/8|
|N2_snr-6|y_pre|13/16|15/16|7/8|
|N2_snr-6|off|12/16|15/16|7/8|
|N2_snr-6|router|12/16|15/16|7/8|
|N1_snr+6|noisy|15/16|15/16|7/8|
|N1_snr+6|y_pre|16/16|16/16|8/8|
|N1_snr+6|off|16/16|16/16|8/8|
|N1_snr+6|router|15/16|15/16|7/8|
|N2_snr+6|noisy|14/16|14/16|6/8|
|N2_snr+6|y_pre|16/16|16/16|8/8|
|N2_snr+6|off|16/16|16/16|8/8|
|N2_snr+6|router|16/16|16/16|8/8|

The Consonant Challenge Corpus, DOI10.5281/zenodo.3538223, CC-BY-4.0. Frozen recipes/source hashes and exact pair seeds are in outputs/recipe_frozen.json. Raw float32 outputs are primary; PCM16 copies are for optional listening. No old experiment was rerun.

Fresh source manifest SHA256: `31c4e4aa9abd9a2cae0cf8b2576942506e3f1cb18cfdb5d7ffd356ff24773ff2`. Speakers f18,m5,f20,m15 exclude all9 speakers in the previous28-file manifest. All16 nominal identities and8 pairs retained. Held out refers to previous project analyses; pretrained exposure is unknown. Retrieval verification remains retrieval/worker_validation.json.

Padded clean gate passed: 16/16 24-way and 16/16 binary; each contrast8/8. All4 frozen noisy conditions executed. The noisy gates tag qualification only: N1_snr-6: False, N2_snr-6: False, N1_snr+6: True, N2_snr+6: True. No input-based sample or condition removal occurred.

Source-recording-support SNR (entire resampled source recording, not phoneme-local or clean-active):

|Condition|Mean dB|Minimum dB|Maximum dB|
|---|---:|---:|---:|
|N1_snr-6|0.108805|-0.737852|0.842933|
|N2_snr-6|0.937676|-0.768228|1.503503|
|N1_snr+6|12.108805|11.262148|12.842933|
|N2_snr+6|12.937676|11.231772|13.503503|

Full-carrier SNR remains exactly the registered ±6dB up to float64 rounding; support SNR uses [8000,8000+resampled_samples), so different clean padding fractions and noise power explain the numerical difference. Per-sample float64 and effective float32 carrier SNR values are in worker_validation.json. These windows do not define true consonant SNR.

Correction and harm counts per16 inputs:

|Condition|Comparison|24-way correct→wrong|24-way wrong→correct|Binary correct→wrong|Binary wrong→correct|
|---|---|---:|---:|---:|---:|
|N1_snr-6|y_pre_versus_baseline|1|7|0|5|
|N1_snr-6|off_versus_baseline|0|7|0|5|
|N1_snr-6|router_versus_baseline|1|7|0|5|
|N1_snr-6|off_versus_y_pre|0|1|0|0|
|N1_snr-6|router_versus_off|1|0|0|0|
|N2_snr-6|y_pre_versus_baseline|1|8|0|2|
|N2_snr-6|off_versus_baseline|2|8|0|2|
|N2_snr-6|router_versus_baseline|2|8|0|2|
|N2_snr-6|off_versus_y_pre|1|0|0|0|
|N2_snr-6|router_versus_off|0|0|0|0|
|N1_snr+6|y_pre_versus_baseline|0|1|0|1|
|N1_snr+6|off_versus_baseline|0|1|0|1|
|N1_snr+6|router_versus_baseline|1|1|1|1|
|N1_snr+6|off_versus_y_pre|0|0|0|0|
|N1_snr+6|router_versus_off|1|0|1|0|
|N2_snr+6|y_pre_versus_baseline|0|2|0|2|
|N2_snr+6|off_versus_baseline|0|2|0|2|
|N2_snr+6|router_versus_baseline|0|2|0|2|
|N2_snr+6|off_versus_y_pre|0|0|0|0|
|N2_snr+6|router_versus_off|0|0|0|0|

Pair collapse and nominal-label agreement (all8 pairs; ties remain errors):

|Condition|Arm|Both 24-way correct /8|Both binary correct /8|24-way collapse /8|Binary collapse /8|
|---|---|---:|---:|---:|---:|
|padded_clean|clean|8|8|0|0|
|N1_snr-6|noisy|2|2|2|6|
|N1_snr-6|y_pre|6|7|0|1|
|N1_snr-6|off|7|7|0|1|
|N1_snr-6|router|6|7|0|1|
|N2_snr-6|noisy|0|5|2|3|
|N2_snr-6|y_pre|6|7|0|1|
|N2_snr-6|off|5|7|0|1|
|N2_snr-6|router|5|7|0|1|
|N1_snr+6|noisy|7|7|1|1|
|N1_snr+6|y_pre|8|8|0|0|
|N1_snr+6|off|8|8|0|0|
|N1_snr+6|router|7|7|1|1|
|N2_snr+6|noisy|6|6|2|2|
|N2_snr+6|y_pre|8|8|0|0|
|N2_snr+6|off|8|8|0|0|
|N2_snr+6|router|8|8|0|0|

Cross-noise stability over16 identities:

|Carrier SNR|Arm|Metric|Prediction changed|Correct in both noises|Stable incorrect|
|---|---|---|---:|---:|---:|
|-6|baseline|24way|10|4|2|
|-6|baseline|binary|7|8|1|
|-6|y_pre|24way|4|12|0|
|-6|y_pre|binary|0|15|1|
|-6|off|24way|4|12|0|
|-6|off|binary|0|15|1|
|-6|router|24way|5|11|0|
|-6|router|binary|0|15|1|
|6|baseline|24way|3|13|0|
|6|baseline|binary|3|13|0|
|6|y_pre|24way|0|16|0|
|6|y_pre|binary|0|16|0|
|6|off|24way|0|16|0|
|6|off|binary|0|16|0|
|6|router|24way|1|15|0|
|6|router|binary|1|15|0|

Original versus fresh speakers, descriptive only (different speakers, no significance inference):

|Condition|Arm|Original 24-way /16|Fresh 24-way /16|Original binary /16|Fresh binary /16|
|---|---|---:|---:|---:|---:|
|N1_snr-6|noisy|8|8|11|10|
|N1_snr-6|y_pre|12|14|13|15|
|N1_snr-6|off|13|15|14|15|
|N1_snr-6|router|15|14|15|15|
|N2_snr-6|noisy|6|6|14|13|
|N2_snr-6|y_pre|12|13|14|15|
|N2_snr-6|off|12|12|14|15|
|N2_snr-6|router|12|12|14|15|
|N1_snr+6|noisy|15|15|15|15|
|N1_snr+6|y_pre|15|16|15|16|
|N1_snr+6|off|15|16|15|16|
|N1_snr+6|router|16|15|16|15|
|N2_snr+6|noisy|16|14|16|14|
|N2_snr+6|y_pre|16|16|16|16|
|N2_snr+6|off|16|16|16|16|
|N2_snr+6|router|16|16|16|16|

Fixed post-padding [1.3,2s] raw RMS medians, separately from identity metrics:

|Condition|Y_pre|Off|Router|
|---|---:|---:|---:|
|N1_snr-6|1.92008957e-05|0.000167062707|0.00011827214|
|N2_snr-6|2.61598556e-05|8.54919471e-05|6.52216182e-05|
|N1_snr+6|1.34319542e-05|5.63551707e-05|4.35151397e-05|
|N2_snr+6|2.05467058e-05|4.95906798e-05|4.06019591e-05|

All source clean padding is exactly zero; output pre[0,0.3s], post[1.3,2s] and outside-source-support RMS/mean-square are verified against cached raw waveforms. Padding energy is not a speech, word or hallucination label. Full ranges are in metrics.json and per-case energies in outputs/audio_manifest.json.

Cache validation passed with zero model forwards:80 clean/noisy recipes regenerate at both float64 and float32 hashes;16 newly resampled clean arrays match originals;320 raw/PCM16 audio pairs match file/waveform hashes and exact installed libsndfile serialization;6528 CTC scores, probabilities and JSON scores recompute with zero error. Summaries, eligibility, transitions and stability reproduce from cached scores. All8 common pair seeds and64 frontend RNG-invariance checks pass. Source recordings, checkpoint, recognizer, core code, frozen protocol and all old±6 records remain unchanged. Runtime uses80kEMA/N30, noisy concatenation=true, cross-attention=false, FiLM=false, selector=none; frontend storm and residual refinement=false checked by driver.

PCM16 copies have 0 clipped samples and maximum quantization difference 3.0517338e-05. One step is1/32768 (−90.309dBFS); recognition uses unclipped float32. Linear energies retain exact zeros; no finite dB floor is inserted. No new model downloads, training or DNSMOS; no human listening result or optional page added.

Exact command: `PATH="/data/coding/sgmse/env/bin:$PATH" /data/coding/sgmse/env/bin/python experiments/p8_q_vcv_speaker_replication_20261019/run_replication.py` from `/data/coding/sgmse`, exit0. Script duration 122.096026s (<600s); external wall/CUDA-only time uninstrumented. Environment: Python3.10.22, torch2.14.1+cu130, CUDA13.0, cuDNN92400, NVIDIA L40S driver610.57.04. Artifacts about149.84MB (<200MB), free disk2.464GB. Preparation metrics/report preserved; original runner metrics at outputs/runner_metrics.json.
