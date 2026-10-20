# Multi-trajectory Distill-MOS audit (2026-10-20)

## Scope

Distill-MOS v7 was evaluated on the 128 WAVs already generated for the 16-file development and 16-file held-out four-trajectory experiments. No new stochastic inference, training, audio copy, dataset access, or checkpoint modification occurred.

The audit compares three selections per file:

- fixed seed 2026;
- maximum DNSMOS-OVRL trajectory;
- the consistency/gain guard: accept the OVRL winner only when its OVRL is higher and its consistency is no worse than seed 2026.

## Decision boundary

Distill-MOS is independent of the DNSMOS-OVRL selection score, but it is still a learned proxy rather than human MOS or a content transcription metric. A positive OVRL gain with no independent Distill-MOS support would be evidence of selection-metric optimism.

## Results

The main paired means are:

| Split / selection | Distill-MOS Δ | OVRL Δ | SI-SDR Δ (dB) | negative SI-SDR files |
|---|---:|---:|---:|---:|
| Dev / OVRL winner | +0.0255 | +0.0548 | +0.0949 | 3/16 |
| Dev / consistency guard | +0.0258 | +0.0442 | +0.0500 | 2/16 |
| Held-out / OVRL winner | **−0.0141** | +0.0646 | +0.0248 | 6/16 |
| Held-out / consistency guard | **−0.0109** | +0.0433 | +0.0244 | 2/16 |

On held-out files, Distill-MOS delta has near-zero correlation with OVRL delta for the OVRL winner (Pearson −0.088) and for the guard (−0.059). The guard reduces SI-SDR regressions, but its independent Distill-MOS delta is still negative.

The exact means, correlations, and per-trajectory scores are in `metrics.json`, `per_trajectory.csv`, and `selected_summary.csv`. The report is intentionally generated from existing WAVs and does not claim a new inference result.

## Decision

The held-out result is consistent with selection-metric gaming or metric mismatch: OVRL rises while an independent learned quality proxy falls. The consistency guard is useful as a risk-ranking diagnostic, but it does not rescue the selector as a perceptual method. Stop the multi-trajectory selector as a current innovation direction; do not train a verifier or run full-test selector inference. Reopen only with an independently validated perceptual endpoint or human preference/transcription evidence.
