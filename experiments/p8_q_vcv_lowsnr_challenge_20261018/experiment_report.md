# VCV independent low-SNR challenge

## Research Lead interpretation

The N1 results contain both improvement and harm. Off changes two previously
correct frontend24-way identities to wrong ones while correcting three;
Router repairs two off errors without another incorrect transition. N2 shows
no Router benefit in correctness counts. Do not generalize N1's two-case
result to all noise types or train a new loss on that small apparent gain.

For N1, f11ata1 is nominal t (frontend t, off k, Router t) and f6ada1 is
nominal d (frontend d, off t, Router d). These are useful candidates for a
later optional human check, not confirmed perceived phonetic corrections.
The user is away; no feedback is awaited and no listening result is assumed.

The carrier SNR must remain explicit. Minus6dB over2s corresponds to about
plus0.87dB on average over the source-recording support, not−6dB within the
consonant. The fixed post-padding energy also illustrates why nominal central
identity agreement and unwanted local sound must be evaluated separately.

Next prioritize a predefined speaker-disjoint VCV replication, retaining the
same noises, carrier SNRs, mappings and scoring. Do not replace current
examples or retune their seeds to increase the observed Router effect.

Status: completed. Frontend/backend/recognizer forwards: 32/64/96.

Independent low-SNR challenge using externally known nominal corpus phoneme identities. Both selected input conditions failed the prior eligibility gate and remain unqualified; the completed +6dB protocol and results are immutable. This challenge is not evidence of ignored noisy identity cues, input identifiability or human recoverability. Recognition failure does not prove that identity evidence is absent. Corpus VCV speech is out of distribution; this is not natural DNS generalization. Nominal identities are corpus labels, not model truth. Padding energy is not a word-hallucination metric. No WER, DNSMOS or population claims.

|Condition|Arm|24-way|Binary|Both correct binary pairs|Binary correct but 24-way wrong|
|---|---|---:|---:|---:|---:|
|N1_snr-6|clean_reused|16/16|16/16|8/8|0/16|
|N1_snr-6|noisy_reused|8/16|11/16|3/8|3/16|
|N1_snr-6|y_pre|12/16|13/16|5/8|1/16|
|N1_snr-6|off|13/16|14/16|6/8|1/16|
|N1_snr-6|router|15/16|15/16|7/8|0/16|
|N2_snr-6|clean_reused|16/16|16/16|8/8|0/16|
|N2_snr-6|noisy_reused|6/16|14/16|6/8|8/16|
|N2_snr-6|y_pre|12/16|14/16|6/8|2/16|
|N2_snr-6|off|12/16|14/16|6/8|2/16|
|N2_snr-6|router|12/16|14/16|6/8|2/16|

Primary recognition uses unclipped raw float32 arrays. PCM16 WAVs are listening copies with clipping counts recorded. Baseline clean/noisy scores are reused without forward passes. Fixed quiet padding RMS and true-support energy are in outputs/audio_manifest.json. Source corpus: The Consonant Challenge Corpus, DOI10.5281/zenodo.3538223, CC-BY-4.0. Optional paired listening page: listening.html. No listening judgment collected.

Full-carrier SNR remains −6dB over the complete2s input. Independently verified SNR over source recording support has mean 0.866557dB, range [-0.330103, 3.194813]dB. The support is [8000,8000+resampled_samples), containing the entire VCV recording; it is not phoneme-local or clean-active SNR. Clean energy is concentrated in this shorter support, explaining the difference without any recipe change. See support_snr_validation.json and the immutable prior support_snr_audit.json. Both original −6dB input gates remain failed; this separate known-label challenge does not requalify them.

Transitions from noisy (per16 inputs):

|Condition|Arm|24-way correct→wrong|24-way wrong→correct|Binary correct→wrong|Binary wrong→correct|
|---|---|---:|---:|---:|---:|
|N1_snr-6|y_pre|0|4|0|2|
|N1_snr-6|off|0|5|0|3|
|N1_snr-6|router|0|7|0|4|
|N2_snr-6|y_pre|1|7|1|1|
|N2_snr-6|off|1|7|2|2|
|N2_snr-6|router|1|7|2|2|

Backend transitions (per16 inputs):

|Condition|Comparison|24-way correct→wrong|24-way wrong→correct|Binary correct→wrong|Binary wrong→correct|
|---|---|---:|---:|---:|---:|
|N1_snr-6|off_versus_y_pre|2|3|1|2|
|N1_snr-6|router_versus_off|0|2|0|1|
|N2_snr-6|off_versus_y_pre|1|1|1|1|
|N2_snr-6|router_versus_off|0|0|0|0|

Pair collapse counts:

|Condition|Arm|Same 24-way prediction /8|Same binary prediction /8|
|---|---|---:|---:|
|N1_snr-6|clean_reused|0|0|
|N1_snr-6|noisy_reused|2|5|
|N1_snr-6|y_pre|2|3|
|N1_snr-6|off|1|2|
|N1_snr-6|router|0|1|
|N2_snr-6|clean_reused|0|0|
|N2_snr-6|noisy_reused|1|2|
|N2_snr-6|y_pre|1|2|
|N2_snr-6|off|0|2|
|N2_snr-6|router|0|2|

Cross-noise stability over16 identical nominal identities:

|Arm|Metric|Same prediction|Correct in both noises|Stable incorrect|Changed prediction|
|---|---|---:|---:|---:|---:|
|y_pre|24way|12|10|2|4|
|y_pre|binary|15|13|2|1|
|off|24way|10|10|0|6|
|off|binary|14|13|1|2|
|router|24way|12|12|0|4|
|router|binary|15|14|1|1|

Fixed post-padding [1.3,2.0s] raw RMS, separately from identity scores:

|Condition|Arm|Minimum|Median|Maximum|
|---|---|---:|---:|---:|
|N1_snr-6|clean|0|0|0|
|N1_snr-6|noisy|0.0192621833|0.0192621833|0.0192621833|
|N1_snr-6|y_pre|1.51027045e-05|1.88182574e-05|0.000388870199|
|N1_snr-6|off|4.00907908e-05|0.000137936234|0.00041571245|
|N1_snr-6|router|3.11042705e-05|9.7781094e-05|0.000285318317|
|N2_snr-6|clean|0|0|0|
|N2_snr-6|noisy|0.0219358948|0.0219358948|0.0219358948|
|N2_snr-6|y_pre|2.39644172e-05|2.84118464e-05|3.80033386e-05|
|N2_snr-6|off|3.0926882e-05|9.24082947e-05|0.000234734695|
|N2_snr-6|router|2.57378736e-05|6.89388544e-05|0.0001626195|

Raw pre-padding and full outside-source-support energy remain in outputs/audio_manifest.json; fixed pre[0,0.3s] clean padding and post padding contain exact zeros. Padding energy measurements do not label generated speech, words or hallucinations.

Validation passed without model forwards: all32 recipe inputs regenerate exactly; all160 finite32000-sample float32 reference/output arrays and their PCM16 WAV serializations match; all2304 enhanced CTC scores and1152 reused baseline CTC scores/probabilities reproduce exactly. Summaries, binary-only cases and noisy transition counts reproduce from cached scores. All frozen source/checkpoint/config/runner/core-code and old+6 artifact hashes remain unchanged. All8 pair seeds are shared across identities/noises/arms; all32 frontends preserve CPU/CUDA RNG state. Runtime flags retain noisy concatenation=true, cross-attention=false, FiLM=false, selector=none, EMA80k/N30.

Primary recognition uses unclipped raw float32. Listening PCM16 clips 0 samples and differs from clipped raw by at most 3.05173414e-05. Conversion is checked using installed libsndfile byte serialization. One quantization step is1/32768 (−90.309dBFS); linear padding RMS/mean-square are retained and exact zero has−infinity dBFS rather than an arbitrary finite floor. No human listening result was collected.

Exact command: `PATH="/data/coding/sgmse/env/bin:$PATH" /data/coding/sgmse/env/bin/python experiments/p8_q_vcv_lowsnr_challenge_20261018/run_challenge.py` from `/data/coding/sgmse`, exit0. Script time 62.623203s (<600s); external process wall/CUDA-only time was not instrumented. Python3.10.22, torch2.14.1+cu130, CUDA13.0, cuDNN92400, L40S, driver610.57.04. No training/download/DNSMOS. Artifact size about59.29MB (<100MB); free disk2.618GB. Preparation records preserved; original runner metrics at outputs/runner_metrics.json. Optional listening.html is available, with no requested or assumed human judgment.
