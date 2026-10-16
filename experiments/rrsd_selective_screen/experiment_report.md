# RRSD selective time-window screen

## Hypothesis

The offline value calibration suggested that Future-grad routing helps mostly at intermediate diffusion times. This experiment tested whether hard enabling RRSD only for `0.15 <= t <= 0.75` improves real enhancement quality.

## Setup

- 80k RRSD EMA checkpoint
- 32 paired utterances from the existing manifest
- Same noisy inputs, per-file seed 2026, PC sampler, `N=30`, `snr=0.5`, no corrector
- Arms: existing `off` (scale 0), existing `router` (scale 1), and new selective window (scale 1, `t_min=0.15`, `t_max=0.75`)
- No training and no dataset/checkpoint modification

## Results

| metric | off | router | selective | selective - router |
|---|---:|---:|---:|---:|
| PESQ | 1.729178 | 1.749041 | 1.740414 | -0.008627 |
| STOI | 0.830958 | 0.831925 | 0.831299 | -0.000627 |
| SI-SDR (dB) | 7.278470 | 7.801027 | 7.468486 | -0.332540 |

Paired bootstrap 95% intervals for selective minus router were `[-0.015585, -0.001346]` for PESQ, `[-0.000897, -0.000346]` for STOI, and `[-0.374578, -0.294202]` dB for SI-SDR. The selective arm was worse on all three primary metrics.

## DNSMOS and conclusion

The selective window improves estimated perceptual quality relative to the full Router:

| metric | selective mean | selective - router | paired bootstrap 95% CI |
|---|---:|---:|---:|
| DNSMOS-SIG | 3.493366 | +0.015659 | [+0.010260, +0.021606] |
| DNSMOS-OVRL | 3.202045 | +0.016646 | [+0.010890, +0.022912] |
| DNSMOS-P808 | 4.041344 | +0.019483 | [+0.012864, +0.026454] |
| DNSMOS-BAK | 4.144153 | +0.001381 | [-0.000899, +0.003753] |

The result is therefore a clear Pareto trade-off, not a failed transfer: hard selective routing improves DNSMOS-SIG/OVRL/P808 while reducing SI-SDR, STOI, and PESQ. This is directly relevant to the project goal of resolving the SI-SDR versus DNSMOS conflict. It should be kept as a candidate perceptual operating point and as evidence for a future learned reliability gate. It should not yet be treated as the final model because the window is hand-designed and the evaluation uses one deterministic seed per utterance.

## Artifacts

Raw selective audio and paired metrics are under `experiment_logs/rrsd_selective_screen/`; the evaluator is `rrsd_selective_paired_eval.py`.
