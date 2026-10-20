# Fixed-noise VCV diagnostic

Status: prepared_no_inference. CPU audio preparation and synthetic validation passed. Recognizer forwards: 0. No enhancement, training, network request or model download was performed.

The existing clean pilot completed at 16/16 for both metrics. This experiment uses the same frozen 16 primary corpus recordings and 8 pairs, with no supplementary samples. Sources and the local checkpoint remain untouched. The runner records and checks their hashes.

Each source is decoded to float64, resampled by scipy.signal.resample_poly(up=16, down=25), placed at sample8000 in a32000-sample zero array and scaled to full2s RMS0.01. The maximum resampled length is fixed at24000. Float64 preserves the prescribed gains and SNR computation; recognition receives a separately hashed float32 conversion. Each clean waveform is reused across every condition. RMS values differ from0.01 by at most2e-18 due to floating point rounding.

N1 and N2 are noise_fileid_5243.wav and noise_fileid_6339.wav from the same DNS source noise directory. Both full originals are verified mono16kHz30s recordings. The fixed5.0–7.0s intervals are normalized to full2s RMS1.0; noise gains0.01*10**(-SNR/20) are constant across all16 identities. SNRs are[-12,-6,0,6]dB. The maximum absolute float64 SNR error is3.56e-15dB. Maximum peak across all144 recipes is0.296539; there is no clipping or postmix scaling. This source input group is a controlled corpus-train VCV plus DNS-source-noise diagnostic, not an official test set.

inputs/recipe_frozen.json freezes source hashes, labels, pairs, gains, mappings, cached-array hashes, all144 waveform hashes, model-input hashes, runner/config/scoring-module hashes and existing checkpoint hashes before any model output. inputs/ contains16 float64 clean NPZ arrays and2 unit-noise NPZ arrays; all recipes regenerate from those arrays without a noisy WAV file. Preparation occupies about1.21MB. The run has a75MB whole-experiment artifact cap.

The runner first scores16 padded clean cases using the same24 exact three-token CTC candidates and mappings as the prior clean calibration. Every candidate is ranked before target evaluation. The gate requires at least12/16 24-way,14/16 binary and6/8 24-way in each contrast. Failure stops noisy inference without padding tuning or sample removal. Passing executes all128 noisy forwards, with no early selection or stopping by results. Float32 logits/probabilities and24 candidate scores are cached per condition, alongside descriptive free greedy runs.

Per-condition summaries include24-way and binary confusion, both-identities-correct pair counts and fractions, same-prediction collapse, contrast counts and margins. Pooled noise/SNR summaries preserve condition-specific pair units. N1/N2 stability includes all16 cases and explicitly reports stable incorrect predictions. A later pilot condition is eligible only if it reaches12/16 24-way,14/16 binary,6/8 24-way for each contrast and6/8 both-correct binary pairs. Pooled results cannot qualify a mixed oracle condition.

AST syntax checking, regeneration/hash checks for144 recipes and synthetic perfect/collapsed pair scoring passed; see preparation_validation.json. No model was initialized during preparation. The recognition output directory is exclusively created and partial/completed runs are refused. RUNNING/FAILED/COMPLETED markers and failure records make interrupted runs visible.

No human recoverability claim is made. Failure to rank the correct identity does not prove that the input contains no identity evidence. No population accuracy, independence, confidence interval or enhancement claims follow from this small controlled diagnostic.

Run once from the repository root:

```bash
env/bin/python experiments/p8_q_vcv_fixed_noise_20261018/fixed_noise.py --device cuda
```

The command updates top-level metrics.json and this report after the prescribed run. The frozen config.yaml remains the preparation protocol snapshot. Do not rerun the preparation or overwrite existing partial outputs.
