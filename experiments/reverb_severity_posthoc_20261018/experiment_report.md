# Continuous reverb-severity post-hoc analysis — 2026-10-18

## Question

The frozen final evaluation showed a Router interaction between `no_reverb` and `with_reverb`. This read-only analysis asks whether that interaction changes continuously with the amount of clean-reference change within each matched file, rather than treating the split label as a complete causal explanation.

## Protocol

- Reused the locked 150 matched file IDs from `final_eval_full_20261018`; no inference was rerun.
- Compared the existing `no_reverb` and `with_reverb` clean references using waveform relative L2, complex-STFT relative L2, log-magnitude STFT L1, and RMS ratio.
- For Router and `t_cut=0.05`, computed the matched-file interaction for each metric:

  `(arm - off)_with_reverb - (arm - off)_no_reverb`.

- Reported Spearman correlations between each clean-change proxy and each interaction, with 10,000 matched-file bootstrap resamples.

## Interpretation boundary

These are post-hoc severity proxies, not measured RT60 or direct estimates of the missing Microsoft RIR. Correlation cannot identify whether the cause is frontend conditioning, phase/reconstruction behavior, training distribution mismatch, or evaluator behavior. The official test manifest was already frozen before this analysis; results must not be used to choose a new threshold on that manifest.

## Results

The clean-change proxies did not provide a stable explanation for the perceptual interaction. For the Router arm, Spearman correlations with the DNSMOS-OVRL interaction were:

| proxy | rho | 95% bootstrap CI |
|---|---:|---:|
| waveform relative L2 | -0.081 | [-0.234, 0.075] |
| complex-STFT relative L2 | -0.081 | [-0.233, 0.076] |
| log-magnitude STFT L1 | -0.098 | [-0.256, 0.067] |
| clean RMS ratio | -0.102 | [-0.256, 0.060] |

All four intervals include zero. The strongest nonzero associations were with SI-SDR interaction: log-magnitude STFT L1 `rho=0.185` (CI `[0.023, 0.338]`) and RMS ratio `rho=0.294` (CI `[0.142, 0.436]`). These are reconstruction-related associations and do not explain the DNSMOS/PESQ degradation. The `t_cut=0.05` arm has the same qualitative result; complete values are in `metrics.json`.

## Decision

The binary reverb interaction remains a reproducible descriptive finding, but these continuous clean-change proxies do not identify a usable severity variable or a perceptual mechanism. Do not design a severity gate from this analysis and do not expand post-hoc analysis on the frozen test set. A causal reverb follow-up would require an independent fixed RIR resource and fresh CV data.

## Reproduction

```bash
env/bin/python3.10 experiments/reverb_severity_posthoc_20261018/analyze_reverb_severity.py
```

`per_file.csv` contains all paired proxies and interaction values; `metrics.json` contains correlations and bootstrap intervals. The DNS Ins20 dataset and all checkpoints were read only.
