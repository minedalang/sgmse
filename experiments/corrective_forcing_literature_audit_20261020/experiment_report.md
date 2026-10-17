# Corrective Forcing literature audit (read-only)

## Source and scope

This audit read the v1 arXiv source of *Corrective Forcing: Unified Post-Training for Diffusions and Flows in Generative Speech Enhancement* (arXiv:2609.24651, submitted 2026-09-21). The source archive SHA-256 is recorded in `config.yaml`. No project data, checkpoint, training code, or inference output was changed.

## What the paper actually proposes

The paper defines post-training directly on recursively generated inference states. Its dynamic rollout correction (DRC) samples a dynamic number of solver steps, generates a partial rollout with the same solver used at inference, stops gradients through the rollout state, and supervises a clean-speech prediction against the clean endpoint. Its counterfactual transition consistency (CTC) additionally compares a factual transition driven by an EMA model with a locally corrected counterfactual transition driven by the ground-truth corrective prediction. The paper uses a shared clean-speech prediction parameterization so the idea can be applied to score/noise predictions after an appropriate conversion.

The paper does **not** compare reverse rollout states with the analytical forward marginal and call that mismatch evidence. This directly confirms the boundary already recorded in the project audit.

## Relevance to this project

The project’s matched-time result (fresh-CV rollout `x0` NMSE `0.5140` versus analytical-forward `0.1356`) belongs to the same train/inference state-mismatch family. The subsequent 16,512-parameter rollout adapter is therefore best interpreted as a small intervention in a method space already occupied by CoF, not as an independent algorithmic contribution. Its proxy improvement and unseen-speaker DNSMOS-OVRL decrease do not overturn that boundary.

## Decision

Do not expand the existing adapter or label it Corrective Forcing. If this line is revisited, the minimum scientifically useful experiment is a faithful DRC baseline for the project’s OUVE PC solver, with recursive rollout states, stop-gradient state capture, clean-`x0` conversion from the score parameterization, and a matched backend-only no-loss control. CTC should not be added until DRC itself is implemented and its final audio metrics show headroom. Any paper contribution would require a StoRM-specific component beyond DRC/CTC, such as a separately validated front-end reliability-conditioned correction; current evidence does not support claiming one.

## Limits

This is a literature and novelty audit, not a reproduction of the paper’s experiments. The paper reports results on SB-VE and OT-CFM rather than this project’s OUVE StoRM checkpoint. Its reported gains cannot be transferred to this dataset or model without a controlled reproduction.
