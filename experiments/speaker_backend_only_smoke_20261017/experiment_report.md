# Backend-only Speaker training-path smoke

## Question

Can Speaker Loss update the NCSN++ diffusion backend while the StoRM frontend and existing RRSD Router remain fixed?

## Configuration

The run initialized from the 80k EMA Router checkpoint, used the existing CV SCP, batch size 8, one CUDA training step, UniSpeech-SAT X-Vector speaker supervision (`contrastive_weight=10`, `contrastive_speaker_weight=0.5`, same-speaker negative masking), `--frontend_trainable` disabled, and the new `--rrsd_freeze_router` flag.

## Result

- Exit code: `0`; one optimizer step completed without NaN or shape errors.
- Lightning reported `27.8M` trainable parameters and `128M` non-trainable parameters.
- The `42.6K`-parameter RRSD Router was frozen; the StoRM frontend was in eval/frozen mode.
- EMA tracked 270 trainable NCSN++ tensors.
- Train loss at the only step: `2683.139`.
- No checkpoint was written (`--nolog`), and no source data was modified.

This verifies the intended backend-only optimization path. It is an engineering smoke, not evidence of a quality improvement. The next experiment must compare the same backend-only budget with and without Speaker Loss before any longer run.


## Objective audit correction (2026-10-17)

The smoke command set `--rrsd_weight 1.0` to instantiate the router and froze its parameters, but `training_step` still adds the RRSD auxiliary loss whenever `rrsd_weight > 0`. Therefore this one-step run verified that gradients and the optimizer can update NCSN++ tensors with the frontend and router frozen, but **did not verify the intended standard score-matching-only objective plus Speaker Loss**. It was not a quality experiment and no checkpoint was retained. Before a matched backend-only comparison, the training path needs an explicit opt-in to keep the frozen router present while disabling the RRSD auxiliary objective. The no-loss control must use the same speaker-balanced minibatch schedule as the Speaker arm.
