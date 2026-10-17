# Corrective-Forcing rollout-state smoke (2026-10-18)

## Question

Recent work on Corrective Forcing argues that diffusion training states sampled from the analytical path can differ from recursively generated inference rollout states. This smoke checks whether the fixed StoRM checkpoint shows a measurable state-distribution deviation on one existing development utterance.

## Protocol

- Fixed 80k checkpoint; checkpoint, manifest, noisy, and clean hashes are recorded in `metrics.json`.
- One existing P7 development utterance; no official test data.
- PC reverse diffusion, `N=30`, no corrector, `snr=0.5`, seed `20261018`.
- A trace callback captured each pre-predictor `x_t` without writing audio.
- For each state, compute `z_t=(x_t-mean_OUVE(clean,Y_pre,t))/std_OUVE(t)`. The complex Gaussian reference for mean `|z_t|^2` is 2. Clean is used only for this offline statistic and never enters the model or sampler.

## Result

All 30 states were captured for the first smoke utterance and four utterances were then audited in `corrective_forcing_rollout_audit4_20261018/`. Across the four utterances, the mean clean-centered standardized power over steps was `1.3778`, while the final-step mean was `3.2545`; the prior-centered control had means `1.3349` and `2.6718`, respectively. All states were finite.

## Interpretation boundary

This is a method audit, not evidence that the model has a Corrective-Forcing-type train/inference mismatch. The forward OUVE marginal describes analytically perturbed clean samples; a reverse PC rollout is not expected to match that distribution at each time step. Differences can arise from reverse dynamics, discretization, model error, prior initialization, and the clean-aware diagnostic definition itself. Therefore this smoke does not trigger Corrective Forcing training. A valid follow-up would need a rollout-consistent target or replay protocol, a no-op/replay control, and a predeclared threshold on a fresh CV set.
