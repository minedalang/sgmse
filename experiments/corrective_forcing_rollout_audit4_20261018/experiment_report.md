# Corrective-Forcing rollout audit on four development utterances (2026-10-18)

## Question

Recent Corrective Forcing work attributes quality loss to a mismatch between analytical training states and recursively generated inference states. We tested whether a simple comparison of captured reverse states against the clean-conditioned forward OUVE marginal could identify such a mismatch in this StoRM checkpoint.

## Protocol

- Fixed 80k checkpoint, EMA, PC reverse diffusion, `N=30`, no corrector, `snr=0.5`, seed `20261018`.
- Four utterances from the existing P7 development manifest; no official test data.
- Captured 30 pre-predictor `x_t` states per utterance without writing audio.
- Computed clean-centered and prior-centered standardized `|z_t|²` statistics. Clean was used only offline.

## Observations

| quantity | mean over four utterances |
|---|---:|
| clean-centered power, all steps | 1.3778 |
| prior-centered power, all steps | 1.3349 |
| clean-centered power, final step | 3.2545 |
| prior-centered power, final step | 2.6718 |

The per-utterance values and hashes are in `metrics.json`; the raw per-step records are `metrics_0.json` through `metrics_3.json`.

## Correct interpretation

These values show that reverse rollout states do not numerically match the clean-conditioned **forward** OUVE marginal. That is expected in general: the forward marginal is a training perturbation distribution, whereas reverse states are produced by the learned reverse process and its discretized sampler. The result therefore does **not** identify a harmful train–inference mismatch, Score failure, or sampling bottleneck.

## Decision

Do not start Corrective Forcing training or a larger state audit based on this statistic. A meaningful CoF study would require a target defined on self-generated rollout states, such as a teacher/ground-truth corrective transition or a controlled replay comparison, plus a fresh-CV protocol and an explicit success threshold. This audit produced no quality claim and did not modify data or checkpoints.
