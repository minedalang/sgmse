# Multi-trajectory consistency/gain guard audit (2026-10-20)

## Question

Can a simple clean-free rule reduce the reconstruction risk of the existing four-trajectory DNSMOS selector without new inference?

## Rule

For each input, start from fixed seed 2026. Find the candidate with maximum DNSMOS-OVRL among seeds 2026--2029. Accept it only when:

1. its OVRL exceeds seed 2026 by at least a fixed threshold; and
2. its trajectory consistency is no worse (lower or equal) than seed 2026.

Consistency is the existing mean normalized log-STFT distance to the other trajectories. The rule uses no clean/reference signal. This is an offline re-selection from already generated trajectories; no new inference or training was performed.

## Held-out result, threshold 0

| Metric | Mean delta vs seed 2026 | Paired bootstrap 95% CI |
|---|---:|---:|
| DNSMOS-OVRL | +0.0433 | [+0.0172, +0.0723] |
| DNSMOS-SIG | +0.0284 | [+0.0079, +0.0534] |
| DNSMOS-BAK | +0.0366 | [+0.0127, +0.0644] |
| SI-SDR | +0.0244 dB | [−0.0119, +0.0620] |
| PESQ | +0.0100 | [−0.0058, +0.0260] |
| STOI | +0.0012 | [−0.0018, +0.0042] |

The rule accepted 7/16 held-out candidates and had 2 files with negative SI-SDR change. The worst SI-SDR change was −0.1541 dB. Without the consistency condition, the four-trajectory selector accepted all 16 and had 6 negative SI-SDR files; its OVRL gain was +0.0646.

## Interpretation and decision

The rule reduces the number of SI-SDR regressions while retaining a statistically positive OVRL/SIG/BAK mean. It does not provide a strict per-file SI-SDR guarantee, and OVRL is also the selection metric, so the quality gains can be optimistic metric selection effects. The result is a useful risk-ranking diagnostic, not a validated deployable innovation.

Do not train a verifier or modify the production sampler from this result. A future selector study would require an independent perceptual endpoint or human preference labels, a pre-registered reconstruction tolerance, and a fresh held-out trajectory set. The current branch remains an offline candidate rather than a new paper claim.

## Reproduction

```bash
/data/coding/sgmse/env/bin/python3.10 \
  experiments/multitrajectory_consistency_gain_guard_20261020/audit_rule.py
```
