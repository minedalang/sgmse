# Goal supplement: generative-backend headroom and conditional calibration

This supplement records the actionable additions from `10-17早上gpt对于后续研究计划的建议.md`. It is subordinate to the active `/goal`; it does not revive the rejected Harm Head, K-step expansion, fixed-scale scans, or Router-only Speaker Loss.

## Research decision

The current Router strategy space is small, but this does not establish that the full NCSN++ output space is small. The next question is whether the StoRM generative backend has a useful, measurable neighborhood around the 80k EMA solution.

## Execution order

1. **Anchor V2 real-audio clean-reference Oracle.** Use the existing 32-utterance repeated-speaker CV manifest. Solve bounded coefficients for `Y_pre - x0_base` and `Y_noisy - Y_pre`, then put the corrected `x0` into the full N=30 reverse trajectory. Run a 4-file smoke first and expand to 32 only if domains and metrics are valid. Do not train the controller before this test.
2. **Multi-trajectory stochastic headroom.** If storage permits, run the unchanged 80k Router for seeds 2026–2029 on 16 files. Compute a clean-reference Oracle under a 0.05 dB SI-SDR budget. Expand to 32 only if the Oracle exceeds +0.01 OVRL; use +0.02 OVRL as the threshold for a meaningful backend search direction.
3. **Conditional NCSN++ residual adapter.** Only if one of the two headroom tests passes, add a zero-init low-rank/residual adapter to a small subset of NCSN++ blocks. Freeze StoRM, Router, and the remaining backbone. Train with a perceptual surrogate only inside a reconstruction/content trust region; independent DNSMOS, PESQ, STOI, SI-SDR and content metrics decide success.
4. **Dynamic diffusion-time adapter.** Test only after a fixed adapter shows a real gain. Stop if the time/state gate collapses or the dynamic model is no better than fixed strength.
5. **Preference distillation and phonological anchor.** Preference labels must first agree across quality metrics; a phonological anchor requires an actual WER/CER headroom diagnosis. Neither is an immediate implementation target.

## Gates and exits

- Anchor Oracle: target Router-relative DNSMOS-OVRL `>= +0.005` with paired bootstrap lower CI `> 0` and SI-SDR `>= -0.05 dB`; otherwise archive the route.
- Multi-trajectory Oracle: stop below `+0.01 OVRL`; only continue toward a deployable verifier near `+0.02 OVRL`.
- Adapter: two training seeds, held-out paired OVRL lower CI `> 0`, SI-SDR `>= -0.05 dB`, and no stable PESQ loss beyond `0.005`. If the surrogate rises while final DNSMOS/PESQ does not, classify it as metric gaming and stop.

## Feasibility triage

The only near-term high-feasibility actions are the two read-only headroom tests:

| Direction | Feasibility now | Why it is actionable | Gate |
|---|---|---|---|
| Anchor V2 real-audio Oracle | High | Existing controller/sampler hooks and synthetic two-direction solver; no new training | Four-file smoke must be numerically valid before the 32-file run |
| Multi-trajectory Oracle | Medium-high | No model change or training; only repeated inference and clean-reference diagnosis | Stop below `+0.01` OVRL; use `+0.02` as the meaningful-search threshold |
| Fixed NCSN++ residual adapter | Medium, conditional | Small zero-init module is implementable, but its value is unknown until headroom is measured | Do not start before Anchor or trajectory headroom passes |
| Dynamic adapter, preference distillation, phonological anchor | Low for the immediate phase | Each needs an additional diagnosis and has a clear failure mode (gate collapse, label conflict, or absent WER/CER gap) | Keep as later branches only |

This triage is the operational interpretation of the attached deep-research report. It does not authorize a long training run by itself; it prioritizes inexpensive evidence first.

## Completed Anchor V2 gate

The 4-file and 32-file N=30 clean-reference Oracle runs are complete. On 32 files, Anchor Oracle versus Router was `+0.448234 dB` SI-SDR but `-0.070951` DNSMOS-OVRL (95% CI `[-0.095117, -0.047702]`) and `-0.011443` PESQ. This fails the perceptual headroom gate, so Anchor V2 controller training is archived rather than started. Detailed evidence is in `experiments/anchor_v2_real_oracle_20261017/`.

## Current evidence that motivates the Speaker audit but does not replace these gates

The independent CAM++ probe (`experiments/speaker_headroom_20261017/`) found Router same-speaker cross-utterance cosine `0.443` versus clean-clean `0.622`, with a larger gap at low SNR. Historical Speaker runs were Router-only (`experiments/speaker_parameter_audit_20261017/`). A new one-step CUDA smoke (`experiments/speaker_backend_only_smoke_20261017/`) now verifies a separate backend-only path with 27.8M NCSN++ parameters trainable and the frontend/Router frozen; it is still not a quality result and requires a matched no-speaker-loss control.

No route in this supplement permits modifying the dataset, overwriting valid checkpoints, or treating clean references as deployable inference inputs.
