# P7-C observable repair probe

## Question and status

This offline analysis asks whether simple deployment-observable measurements from a noisy one-second chunk and its Frontend-only `Y_pre` predict whether the existing Router-off StoRM output improves or worsens local waveform reconstruction relative to `Y_pre`.

**Status: completed exploratory development-split analysis.** The 32-item P7-C development split has been screened repeatedly. It is not held-out evidence and this analysis is not confirmatory.

## Data and fixed analysis

The analysis paired same-utterance, 16 kHz aligned clean, noisy, `Y_pre`, and Router-off WAVs using `experiments/storm_error_attribution_p7_20261017/baseline_manifest.csv`. The manifest SHA-256 was `77b64db9da6a1ff372a22dcda4a6316743133f383103aaffb3dbce2c671c178e`. All 32 noisy, clean, `Y_pre`, and Router-off files were present; every file hash was checked against the manifest or existing `frontend_only/metrics.json`. Sample rates and mono sample lengths were verified.

Each file was divided into fixed, non-overlapping 1.0-second chunks. There were 960 complete chunks; 5 were excluded because clean RMS was below `1e-4`, leaving **955 chunks from 32 utterances and 30 speakers**. No incomplete tail samples were dropped. Speaker IDs were parsed and validated from the manifest `uttid`; two readers contribute two utterances each and the remaining 28 contribute one each.

The target is

`log(NRMSE(Y_pre, clean)) - log(NRMSE(Router-off, clean))`,

where waveform NRMSE is `sqrt(mean((estimate-clean)^2)) / sqrt(mean(clean^2))`. Positive values mean Router-off has lower reconstruction error than `Y_pre`. Clean audio was used only for this offline target and the prespecified clean-RMS exclusion.

The fixed 12-feature set used only noisy and `Y_pre`: their log RMS values, RMS ratio and difference, each signal's energy fractions in 0–1, 1–4, and 4–8 kHz bands, and each signal's spectral flatness. Band energies use a one-sided real FFT with DC and Nyquist counted once. The feature set was not changed after examining targets. Router-off output was not provided to feature extraction.

The baseline predicts the mean target in each training fold. Ridge regression uses `alpha=1.0`; feature standardization is fit separately on each training fold. Evaluation used speaker-grouped 5-fold cross-validation, with no tuning. The reported 95% intervals use 2,000 speaker-cluster bootstrap resamples of the fixed grouped out-of-fold predictions (seed `20261017`), keeping all chunks from a sampled speaker together.

## Results

The target distribution was mean **-0.0785**, sample SD **0.1598**, median **-0.0809**, p05 **-0.2734**, p95 **0.1208**, minimum **-1.1001**, and maximum **1.1239**. Router-off helped on 200/955 chunks (**20.94%**) and worsened the proxy on 755/955; this segment fraction is descriptive and does not treat chunks as independent confirmation.

| Predictor | Grouped OOF R² | Grouped OOF MAE | Grouped OOF Spearman |
|---|---:|---:|---:|
| Training-fold mean | -0.0056 | 0.0975 | -0.0918 |
| Standardized Ridge (`alpha=1`) | -0.0258 | 0.1050 | 0.0719 |

Ridge performed worse than the mean-only baseline on MAE by 0.0075; its weakly positive rank correlation did not accompany positive OOF R². The speaker-bootstrap 95% interval for Ridge-minus-mean MAE was **[0.0021, 0.0149]**, excluding zero. The interval for Ridge-minus-mean R² was **[-0.1627, 0.0690]**, spanning zero. Individual-model intervals and Spearman intervals are recorded in `metrics.json`.

This small fixed linear probe did not predict local NRMSE changes usefully on these grouped folds. It does not establish that all observable reliability signals or richer/independently tested mechanisms cannot work.

## Interpretation limits

- This is a repeatedly screened 32-item development split, so results are exploratory, not held-out or confirmatory.
- The 955 chunks are clustered within 32 utterances and 30 speakers. Chunk count is not the number of independent observations; bootstrap intervals resample speakers, but still use fixed cross-validation predictions and are approximate.
- The model is a small prespecified linear probe without hyperparameter search. Its result is not a universal test of observable reliability.
- Waveform NRMSE is a local reconstruction proxy only. No DNSMOS or local perceptual claim is made.
- No training or inference was run; the analysis only reused existing WAVs. Datasets and checkpoints were not modified.

## Reproduction and checks

From `/data/coding/sgmse`, the analysis command was:

```bash
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 env/bin/python3.10 experiments/p7c_observable_repair_probe_20261017/analyze_observable_repair.py
```

The script completed in **4.31 seconds** (`analysis_runtime_seconds` in `metrics.json`). Python bytecode compilation, YAML parsing, and JSON parsing succeeded. All 32 paired input WAV hashes matched their recorded values. `CUDA_VISIBLE_DEVICES` was empty and the analysis ran CPU-only; no GPU process was used. There was no dataset or checkpoint write.

Artifacts:

- `config.yaml` — fixed protocol and provenance
- `metrics.json` — data validation, per-file input SHA-256 values, grouping, fold metrics, clustered intervals, target distribution, and descriptive counts
- `analyze_observable_repair.py` — reproducible CPU analysis
- `per_chunk_predictions.csv` — eligible targets, prespecified features, folds, and out-of-fold predictions
