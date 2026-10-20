# VCV enhancement follow-up

## Research Lead assessment and scope of the next experiment

The qualified-input positive control succeeds: the full pipeline generally
follows the nominal recorded identity under both fixed noises. Off preserves
Y_pre's correctness on this set; Router corrects one N1 d/t label error. This
is not evidence of a general Router content benefit or low-SNR recovery.
The output's quiet padding can still contain energy despite high identity
agreement, so correct central identity and absence of local artifacts must
remain separate measurements.

The next−6dB challenge is a separate registered experiment on externally
labeled recordings, not a retrospective relaxation of this stage's input
eligibility gate. Known nominal corpus identities permit reporting enhanced
label agreement even when the noisy recognizer is unreliable. That challenge
cannot use its failed input recognizer gate to claim inputs were unidentifiable,
nor use a correctly guessed output to prove evidence-based recovery. Do not
pool it into this qualified-input positive control or replace its examples.

Status: completed. Frontend/backend/recognizer forwards: 32/64/96.

Recognizer-conditioned positive control selected by the preregistered input gate; not an unbiased enhancement benchmark or a low-SNR recovery claim. VCV corpus speech is out of distribution and +6dB is outside the historical nominal training SNR range. Nominal corpus identities are known labels, not model truth. Padding energy is not a word-hallucination metric. No human recoverability, WER, DNSMOS or population claims.

|Condition|Arm|24-way|Binary|Both correct binary pairs|
|---|---|---:|---:|---:|
|N1_snr+6|clean_reused|16/16|16/16|8/8|
|N1_snr+6|noisy_reused|15/16|15/16|7/8|
|N1_snr+6|y_pre|15/16|15/16|7/8|
|N1_snr+6|off|15/16|15/16|7/8|
|N1_snr+6|router|16/16|16/16|8/8|
|N2_snr+6|clean_reused|16/16|16/16|8/8|
|N2_snr+6|noisy_reused|16/16|16/16|8/8|
|N2_snr+6|y_pre|16/16|16/16|8/8|
|N2_snr+6|off|16/16|16/16|8/8|
|N2_snr+6|router|16/16|16/16|8/8|

Primary recognition uses unclipped raw float32 arrays. PCM16 WAVs are listening copies with clipping counts recorded. Baseline clean/noisy scores are reused without forward passes. Fixed quiet padding RMS and true-support energy are in outputs/audio_manifest.json. Source corpus: The Consonant Challenge Corpus, DOI10.5281/zenodo.3538223, CC-BY-4.0. Optional paired listening page: listening.html. No listening judgment collected.

Per-condition transitions from noisy (24-way and binary counts coincide here):

| Condition | Arm | Correct→wrong /16 | Wrong→correct /16 | Collapse 24-way /8 pairs | Collapse binary /8 pairs |
|---|---|---:|---:|---:|---:|
| N1_snr+6 | y_pre | 1 | 1 | 1 | 1 |
| N1_snr+6 | off | 1 | 1 | 1 | 1 |
| N1_snr+6 | router | 0 | 1 | 0 | 0 |
| N2_snr+6 | y_pre | 0 | 0 | 0 | 0 |
| N2_snr+6 | off | 0 | 0 | 0 | 0 |
| N2_snr+6 | router | 0 | 0 | 0 | 0 |

Backend comparisons (24-way and binary counts coincide):

| Condition | Comparison | Correct→wrong /16 | Wrong→correct /16 |
|---|---|---:|---:|
| N1_snr+6 | off_versus_y_pre | 0 | 0 |
| N1_snr+6 | router_versus_off | 0 | 1 |
| N2_snr+6 | off_versus_y_pre | 0 | 0 |
| N2_snr+6 | router_versus_off | 0 | 0 |

Cross-noise stability, each metric over all16 identities:

| Arm | Metric | Same prediction | Correct in both noises | Stable incorrect |
|---|---|---:|---:|---:|
| y_pre | 24way | 15 | 15 | 0 |
| y_pre | binary | 15 | 15 | 0 |
| off | 24way | 15 | 15 | 0 |
| off | binary | 15 | 15 | 0 |
| router | 24way | 16 | 16 | 0 |
| router | binary | 16 | 16 | 0 |

Worker verification passed with zero model forwards: all32 regenerated input hashes, all160 raw references/outputs and PCM16 byte serializations, all96×24 enhanced CTC scores and48×24 reused baseline CTC scores. Recomputed scores/probabilities have exact zero error; summaries and transitions reproduce from cached data. All source/checkpoint/protocol/core hashes remain unchanged. Fixed noise waveform/gain match within every pair; all8 pair seeds are shared across identities, both noises and backend arms. All32 frontends preserve CPU/CUDA RNG state. Every output has32000 finite float32 samples. Runtime flags match noisy concatenation=true, cross-attention=false, FiLM=false, selector=none.

Actual float64 input SNR ranges from 5.999999999999996 to 6.000000000000001 dB. Clean input pre[0,0.3s] and post[1.3,2s] regions contain exact zeros. All stored padding RMS/mean-square and source-support intervals reproduce exactly from raw outputs; per-condition/arm min/median/max raw padding RMS are in metrics.json. These energies do not assign speech or hallucination labels.

PCM16 provenance: listening copies are produced by the installed soundfile/libsndfile conversion of clipped raw float32 at16kHz and all160 WAV byte sequences reproduce. Total clipped samples: 0; maximum absolute quantization difference: 3.0517333e-05. One PCM16 step is1/32768 (-90.308999dBFS). Primary recognition uses raw float32. Linear padding RMS is retained; exact zero maps to−infinity dBFS, with no invented finite floor. Initial worker validation assumed ideal mathematical floor conversion; one near-integer rounding mismatch exposed that validator assumption. The corrected check reproduces libsndfile bytes; inference was never rerun. Initial validator traceback remains worker_validation_attempt01.log.

Run command: `PATH="/data/coding/sgmse/env/bin:$PATH" /data/coding/sgmse/env/bin/python experiments/p8_q_vcv_enhance_follow_20261018/run_follow.py` from `/data/coding/sgmse`, exit0. Script duration 64.458771s (<600s); external process wall/CUDA-only time were not instrumented. Environment: Python3.10.22, torch2.14.1+cu130, CUDA13.0, cuDNN92400, L40S, driver610.57.04. No downloads, training or DNSMOS. Artifacts approximately59.22MB (<100MB); free disk2.681GB. Original runner metrics are preserved at outputs/runner_metrics.json; preparation records remain preparation_metrics.json/preparation_report.md. Optional listening.html is ready for tomorrow; no listening judgment has been collected.
