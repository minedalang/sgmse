# Paper-level claim audit (2026-10-18)

> **后续证据修正（2026-10-19）：** matched-time rollout x0 gap 之后完成的 rollout-aware backend adapter pilot 曾在 utterance-level holdout proxy 上改善 `42.4%`，但其中 6 条音频参与 adapter 训练、仅 2 条为 holdout；DNSMOS-P808 改善而 DNSMOS-OVRL 略降。随后在独立 4-speaker/8-utterance 子集上复现了 proxy 改善 `54.8%`，但 2 条 unseen-speaker holdout 的 DNSMOS-OVRL 下降 `0.03465`。因此该路线仍是“proxy 复现、感知泛化未确认”的候选机制，不能写成已验证算法创新或 Corrective Forcing 成功；除非提出新的监督假设，不再扩大训练。详细记录见 `experiments/rollout_adapter_pilot_20261018/`、`experiments/rollout_adapter_replication_20261018/`。

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
| Matched-time rollout x0 gap | **Positive diagnostic candidate** | On 8 fresh-CV utterances, rollout-state x0 NMSE `0.5140` versus analytic-forward `0.1356`, difference `+0.3785`, positive for 8/8 | This is not yet a quality result or proof of CoF benefit; a bounded rollout-aware pilot needs detached replay targets and fresh-CV audio metrics | `rollout_prediction_gap_20261018` |
| Rollout-aware backend adapter | **Mixed-evidence candidate; replication complete** | Stable 16.5k-parameter adapter reduced the original 2-utterance holdout x0 NMSE by 42.4%, and an independent 2-utterance unseen-speaker holdout reduced it by 54.8%; DNSMOS-OVRL decreased on the independent holdout by 0.03465 | Not a confirmed perceptual gain, generalization result, Corrective Forcing method, or paper-level innovation; no further expansion without a new supervision hypothesis | `rollout_adapter_pilot_20261018`, `rollout_adapter_replication_20261018` |
| Independent rollout adapter replication | **Proxy replicated; perceptual generalization unresolved** | On a new 4-speaker CV subset (8 utterances), 2 unseen-speaker holdout x0 NMSE decreased 54.8%; holdout DNSMOS-OVRL decreased 0.03465 despite small SI-SDR/STOI gains | Still too small and mixed for a quality claim, Corrective Forcing claim, or paper innovation; stop expanding this route without a new supervision hypothesis | `rollout_adapter_replication_20261018` |
| WavLM harm Oracle DNSMOS read-only check | **Stopped negative/neutral diagnostic** | On four existing `N=30` full/gated WAV pairs, Gate−Full DNSMOS P808/SIG/BAK/OVRL was `−0.000769/−0.000630/−0.000899/−0.000953`; all exploratory utterance-bootstrap intervals crossed zero | No evidence of usable perceptual headroom for a Harm Head; this small sample does not invalidate every possible harm label | `rrsd_harm_oracle_n30_4_dnsmos_20261019` |
| H4 output-level distortion proxy diagnostic | **Stopped exploratory diagnostic** | On 32 matched `off/router/cut005` outputs, harmonicity had moderate associations with some STOI/SI-SDR contrasts, but no proxy repeated coherently across both contrast families and DNSMOS-OVRL | No validated distortion label, causal artifact, or basis for GAN/conditional-discriminator training | `h4_distortion_diagnostic_20261018` |
| Corrective Forcing literature boundary | **Prior-art boundary** | arXiv:2609.24651 defines dynamic rollout correction and counterfactual transition consistency on recursively generated states | A rollout-aware adapter or direct CoF reproduction cannot be claimed as a second innovation; a StoRM-specific contribution would need an independently validated additional mechanism | `corrective_forcing_literature_audit_20261020` |

## Paper wording recommended by the current evidence

The strongest defensible contribution is an empirical and diagnostic result:

> Future guidance provides a small reconstruction benefit with a reproducible perception–reconstruction tradeoff. On the locked DNS Ins20 synthetic evaluation, its effect depends strongly on reverberation: no-reverb utterances gain in PESQ/STOI/SI-SDR, whereas with-reverb utterances lose PESQ/STOI/DNSMOS quality while retaining a small SI-SDR gain. A bounded call-chain, numerical, and observable-repair audit did not identify an independently actionable diffusion bottleneck or a deployable local gate.

This can support a paper section on **condition-dependent limitations and mechanism diagnosis**. It should not be presented as a second new algorithmic innovation.

## Claims explicitly rejected

Do not write that:

1. P7 validated a new diffusion SDE or backend mechanism.
2. Router improves overall perceptual quality on the official test set.
3. DNSMOS-only trajectory selection is a reliable perceptual verifier.
4. Anchor V2, Harm Head, Speaker loss, GAN, or perceptual loss has an independent positive result; the rollout-aware adapter remains mixed evidence despite proxy replication and is not a confirmed perceptual method.
5. H4 harmonicity correlations identify a validated audible artifact or justify adversarial training.
6. Mixed amplitude/phase Oracle results are deployable gains.
7. The ODE tolerance ladder proves the default PC sampler is the bottleneck.
8. The reverb interaction proves a reverb-only causal explanation.
9. A single listener's 24 ties establish perceptual equivalence.
10. Clean-aware Oracle results are evidence available at inference time.

## Decision

P7 core diagnostics are complete as a bounded diagnosis and negative-screening stage. The rollout-aware adapter replication is now complete: its proxy improvement was reproduced, but the unseen-speaker holdout DNSMOS-OVRL decreased, so it remains archived mixed evidence and is not expanded without a new supervision hypothesis. The Harm Oracle DNSMOS check is also complete and does not justify training a Harm Head. The Router/off/t-cut table remains frozen, as does the official test set.
