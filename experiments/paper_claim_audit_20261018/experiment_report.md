# Paper-level claim audit (2026-10-18)

## Purpose

This is a read-only audit of claims that can be made from the completed StoRM experiments. It does not add training, inference, threshold tuning, or test-set selection. The goal is to prevent the P7 diagnosis and small metric changes from being written as a validated new diffusion mechanism.

## Evidence matrix

| Evidence | Status | Claim that is supported | Claim that is not supported | Primary record |
|---|---|---|---|---|
| Frozen official evaluation | **Safe project observation** | Future Router changes the quality tradeoff: pooled SI-SDR improves `+0.241412 dB`, while STOI and DNSMOS-OVRL decrease; the effect differs between no-reverb and with-reverb | No overall perceptual improvement; no claim that Router is superior on the full test set | `final_eval_full_20261018` |
| Matched reverb interaction | **Mechanistic observation with limits** | Router effect interacts with reverberation; matched interaction is PESQ `-0.04110`, SI-SDR `-0.15524 dB`, OVRL `-0.02854` | Reverb is not proven as the sole cause; no justified reverb gate or severity controller | `final_eval_reverb_interaction_20261018` |
| P7-0 call-chain audit | **Implementation evidence** | Router-off still contains StoRM `Y_pre`; PC/N=30 and the single-file replay are reproducible | No evidence of a code-path bug or proof of a defective score model | `storm_error_attribution_p7_20261017` |
| P7-A ODE tolerance ladder | **Negative diagnostic** | A single controlled ODE trace is tolerance-sensitive and expensive at strict tolerances | No convergence proof, no PC-sampler diagnosis, and no basis for changing the SDE | `storm_error_attribution_p7_20261017` |
| P7-B score stratification | **Gate not triggered** | No stable time/frequency Score anomaly was established before an expensive DSM study | Cannot claim Score learning is correct or incorrect in every region | `RESEARCH_PLAN.md`, P7 decision table |
| P7-C local repair probe | **Negative diagnostic** | Router-off helps only 200/955 local waveform-proxy chunks; 12 simple observable features did not predict the gain (Ridge MAE `0.1050` vs mean `0.0975`) | No evidence that no richer observable gate can work; no deployable local gate is validated | `p7c_observable_repair_probe_20261017` |
| Anchor V2 clean Oracle | **Upper-bound/negative route** | A clean-aware residual oracle has SI-SDR headroom (`+0.448234 dB`) but OVRL decreases (`-0.070951`) | No evidence an observable Anchor controller improves perception | `anchor_v2_real_oracle_20261017` |
| Multi-trajectory consistency | **Risk-ranking evidence** | Consistency penalty improves held-out OVRL ranking, but SI-SDR CI crosses zero and negative SI-SDR cases remain | No reliable Pareto verifier; no reason for four-trajectory official inference or verifier training | `multitrajectory_consistency_verifier_20261018` |
| Backend no-loss control | **Stopped candidate** | Ordinary backend-only adaptation did not show speaker gain and caused small PESQ/STOI loss | No basis for Speaker-loss, Harm Head, or backend adapter continuation | `speaker_backend_only_matched_500_20261018` |
| ASR content smoke | **Headroom negative/weak** | Router did not show content improvement over off in the 8-utterance smoke (`+0.00516` pseudo-WER) | No basis for phonological anchor or content-loss training | `content_headroom_asr_smoke_20261018` |
| Perceptual representation probe and listening pilot | **Insufficient supervision** | Simple representations prefer DNSMOS and reconstruction metrics differently; the single listener chose tie in all 24 trials | No human-validated perceptual target and no basis for perceptual/GAN training | `perceptual_representation_probe_20261017`, `p3b_blind_listening_pilot_20261017` |
| Residual-Anchored OUVE audit | **Novelty boundary** | The implemented transform is equivalent to a standard OUVE under its stated fixed-condition assumptions | Cannot present it as a new stochastic dynamics without a separate non-equivalent formulation and evidence | `residual_anchor_equivalence_audit_20261018` |
| Corrective-Forcing rollout audit | **Diagnostic method rejected** | Four captured reverse trajectories differ from the clean-conditioned forward marginal, with final-step standardized power around `3.25` (clean-centered) and `2.67` (prior-centered) | This does not identify train–inference mismatch because reverse states are not expected to follow the forward perturbation marginal; no CoF training is justified | `corrective_forcing_rollout_audit4_20261018` |

## Paper wording recommended by the current evidence

The strongest defensible contribution is an empirical and diagnostic result:

> Future guidance provides a small reconstruction benefit with a reproducible perception–reconstruction tradeoff. On the locked DNS Ins20 synthetic evaluation, its effect depends strongly on reverberation: no-reverb utterances gain in PESQ/STOI/SI-SDR, whereas with-reverb utterances lose PESQ/STOI/DNSMOS quality while retaining a small SI-SDR gain. A bounded call-chain, numerical, and observable-repair audit did not identify an independently actionable diffusion bottleneck or a deployable local gate.

This can support a paper section on **condition-dependent limitations and mechanism diagnosis**. It should not be presented as a second new algorithmic innovation.

## Claims explicitly rejected

Do not write that:

1. P7 validated a new diffusion SDE or backend mechanism.
2. Router improves overall perceptual quality on the official test set.
3. DNSMOS-only trajectory selection is a reliable perceptual verifier.
4. Anchor V2, Harm Head, Speaker loss, adapter, GAN, or perceptual loss has an independent positive result.
5. Mixed amplitude/phase Oracle results are deployable gains.
6. The ODE tolerance ladder proves the default PC sampler is the bottleneck.
7. The reverb interaction proves a reverb-only causal explanation.
8. A single listener's 24 ties establish perceptual equivalence.
9. Clean-aware Oracle results are evidence available at inference time.

## Decision

P7 is complete as a bounded diagnosis and negative-screening stage. The current results justify freezing the Router/off/t-cut table and documenting the reverb interaction. Starting another long training run would require a new independent mechanism hypothesis, a literature novelty check, a fresh-CV or held-out protocol, and a predeclared success/stop rule.
