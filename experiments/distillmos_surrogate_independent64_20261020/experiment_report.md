# Distill-MOS independent CV surrogate check

## Scope

Read-only evaluation of the already completed 64-speaker/128-utterance CV experiment. The existing `off/` and `router/` output WAVs and the paired quality CSV were reused; no inference, training, audio copying, dataset access, or checkpoint modification occurred. Distill-MOS v7 was run on 256 existing output WAVs.

## Corrected protocol and failed first attempt

The first script read `inputs/off` and `inputs/router`, which are staging symlinks rather than enhanced output directories. That produced a constant zero Distill-MOS delta and all-NaN correlations. The attempt is retained in `run.log` and is excluded. The corrected run used `experiment_logs/speaker_headroom_replication_64_20261017/off/` and `/router/`, verified by the nonzero output deltas below.

## Results

| paired delta | mean | Distill-MOS delta correlation (Pearson / Spearman) |
|---|---:|---:|
| SI-SDR | +0.5115 dB | −0.304 / −0.390 |
| PESQ | +0.0109 | +0.049 / −0.020 |
| STOI | +0.00115 | +0.023 / −0.079 |
| DNSMOS-OVRL | −0.0298 | +0.324 / +0.450 |
| DNSMOS-SIG | −0.0309 | +0.330 / +0.457 |
| Distill-MOS | −0.0279 | — |

The independent split does not show a strong positive relationship between Distill-MOS improvement and Router reconstruction gains. Its relationship with DNSMOS deltas is moderate in the opposite setting from a useful reward: Router's mean DNSMOS falls while Distill-MOS also falls, and per-file correlations are not sufficient to establish a deployable quality target. Together with the 32-utterance ranking audit, this does not justify optimizing Distill-MOS as a standalone perceptual reward.

## Decision

Keep the perceptual trust-region adapter closed. A future quality-target route would require an independently validated multi-objective target or human preference data; no further Distill-MOS experiments are planned on the current development material.
