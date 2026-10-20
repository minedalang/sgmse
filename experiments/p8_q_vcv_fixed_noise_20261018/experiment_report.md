# Fixed-noise VCV diagnostic

## Research Lead decision after the completed run

Only N1+6dB and N2+6dB satisfy every preregistered condition-specific gate.
N2 at0dB has14/16 binary correct but only5/8 t/d correct among24 candidates;
do not silently relax that gate or qualify a pooled condition. Binary success
and full-class identity agreement measure different tasks.

The next bounded diagnostic retains all16 recordings in each eligible
condition and compares frontend/off/Router. Common random numbers within
each matched pair and across noise instances will separate speech identity
changes from seed changes. Conditions were selected by the predefined input
recognizer gate, so resulting enhancement scores describe a recognizer-
conditioned positive control, not an unbiased benchmark. Plus6dB and isolated
VCVs also differ from the original DNS training distribution.

Noisy recognition failures at lower SNR limit this recognizer as a recovery
judge. They do not prove information-theoretic non-identifiability or lack of
human acoustic evidence. Further evaluation must not call inserted labels
hallucinations solely because this recognizer's preferred candidate differs.

Status: completed. One invocation; 144 recognizer forwards (16 padded clean plus 128 noisy). Script runtime 13.239932s. No enhancement or training.

Padded clean gate passed: 16/16 24-way, 16/16 binary, k/g 8/8, t/d 8/8. Original primary clean was also 16/16. The historical supplementary 7/12 result remains descriptive only; no supplementary sample was used.

| Condition | 24-way /16 | Binary /16 | k/g /8 | t/d /8 | Both binary correct /8 pairs | Collapse 24-way /8 | Collapse binary /8 | Eligible |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| padded_clean | 16 | 16 | 8 | 8 | 8 | 0 | 0 | — |
| N1_snr-12 | 5 | 8 | 3 | 2 | 0 | 3 | 8 | no |
| N1_snr-6 | 8 | 11 | 4 | 4 | 3 | 2 | 5 | no |
| N1_snr+0 | 12 | 13 | 6 | 6 | 5 | 3 | 3 | no |
| N1_snr+6 | 15 | 15 | 7 | 8 | 7 | 1 | 1 | yes |
| N2_snr-12 | 1 | 9 | 0 | 1 | 1 | 1 | 7 | no |
| N2_snr-6 | 6 | 14 | 3 | 3 | 6 | 1 | 2 | no |
| N2_snr+0 | 12 | 14 | 7 | 5 | 6 | 2 | 2 | no |
| N2_snr+6 | 16 | 16 | 8 | 8 | 8 | 0 | 0 | yes |

Only N1 +6 dB and N2 +6 dB satisfy every predefined later-pilot criterion. N1 0 dB misses binary and pair criteria; N2 0 dB misses the t/d 24-way criterion. No condition was selected to change this experiment.

| SNR dB | Same 24-way /16 | Both noises 24-way correct /16 | Stable incorrect 24-way /16 | Same binary /16 | Both noises binary correct /16 | Stable incorrect binary /16 |
|---|---:|---:|---:|---:|---:|---:|
| -12 | 3 | 0 | 3 | 15 | 8 | 7 |
| -6 | 6 | 4 | 2 | 11 | 10 | 1 |
| 0 | 9 | 9 | 0 | 11 | 11 | 0 |
| 6 | 15 | 15 | 0 | 15 | 15 | 0 |

Every condition uses the same frozen clean waveform plus its fixed noise waveform at a gain identical across both pair identities and all speakers. No clipping, postmix scaling, sample replacement or parameter tuning occurred. All 144 float64 recipes and float32 model inputs regenerate with exact sealed hashes. The largest float64 SNR error is 3.56e-15 dB. After mixture conversion to float32, the maximum effective SNR change is 2.07e-8 dB; per-condition ranges and per-sample values are in worker_validation.json. Effective residual noise is float32 mixture minus separately converted float32 clean, evaluated in float64. The feature extractor subsequently applies its saved preprocessing.

All 144 cached logits arrays were checked without model inference: 3,456 CTC candidate scores, saved probabilities and JSON scores reproduce with maximum absolute error 0. Source recordings, checkpoint, manifest, scoring module, runner, config and recipe hashes remain unchanged. Recipe SHA256: `8936e3edc3f337356cba7943745a9ed2a9939a76e7ce0da4afd21eb84749d769`.

Exact execution command:

```bash
PATH="/data/coding/sgmse/env/bin:$PATH" env/bin/python experiments/p8_q_vcv_fixed_noise_20261018/fixed_noise.py --device cuda
```

Environment: Python 3.10.22, torch 2.14.1+cu130, CUDA build 13.0, cuDNN 92400, NVIDIA L40S, driver 610.57.04. Script runtime includes loading, hashing, forwards, scoring and writing; process wall and CUDA-only duration were not measured. The command exited 0.

Preparation records are preserved in preparation_metrics.json, preparation_report.md and preparation_validation.json. Frozen protocol remains config.yaml; sealed inputs remain inputs/recipe_frozen.json. Original runner output remains recognition_outputs/metrics.json; consolidated metrics and validation are top-level metrics.json and worker_validation.json. Reproduce cached validation with `env/bin/python experiments/p8_q_vcv_fixed_noise_20261018/worker_validate.py` (zero recognizer forwards).

Artifacts currently occupy about 43.50 MB against the 75 MB cap; free disk after validation was 2.741 GB.

Controlled recognizer diagnostic with nominal corpus labels. No human recoverability claim. Failure to rank the correct identity does not prove that the input contains no identity evidence. No population, independence, confidence interval or enhancement claims. Pooled conditions are descriptive; eligibility applies only to a single preset noise/SNR condition, never to oracle mixtures.
