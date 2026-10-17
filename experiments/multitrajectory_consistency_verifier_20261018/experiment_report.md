# Consistency-penalized multi-trajectory verifier — 2026-10-18

## Question

The existing four-trajectory selector chooses the highest DNSMOS-OVRL candidate and improves held-out OVRL, but it has six SI-SDR-negative files out of 16. This read-only analysis tests one fixed no-reference score that combines the candidate's within-utterance DNSMOS-OVRL rank with an inter-trajectory consistency penalty.

## Protocol

- Reused the existing 16 development and 16 held-out utterances and four saved trajectories per utterance.
- No audio was regenerated and no DNSMOS inference was rerun.
- For each utterance, standardize the four candidate OVRL values and consistency values separately, then select

  `argmax [ z(OVRL) - 0.5 z(consistency) ]`.

- The coefficient `0.5` was fixed before reading the summary of this analysis; it was not tuned on held-out files.
- Metrics are reported relative to seed 2026. Bootstrap resamples are utterance-level.

## Results and limits

The fixed score reduced held-out SI-SDR-negative files from 6/16 for the pure OVRL selector to 4/16, while retaining a held-out OVRL delta of `+0.05875` (95% CI `[+0.03265,+0.08651]`). Its held-out SI-SDR delta was `+0.02107 dB`, but the CI `[-0.02697,+0.06907]` crossed zero. PESQ also crossed zero (`-0.00350`, CI `[-0.04005,+0.02373]`). Therefore this fixed verifier is a useful risk-ranking diagnostic, not a validated Pareto controller.

The complete means and intervals are in `metrics.json`. This is a post-hoc exploratory screen on a small, repeatedly used trajectory split. The selection score still uses DNSMOS as a component, so it is not independent human-perception validation. It cannot justify a full test-set run or a learned verifier by itself.

## Reproduction

```bash
env/bin/python3.10 experiments/multitrajectory_consistency_verifier_20261018/analyze_verifier.py
```
