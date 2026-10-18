# Claim matrix for the current StoRM paper

| Claim | Evidence that can be cited | Safe wording | Boundary / excluded wording |
|---|---|---|---|
| Future guidance changes reconstruction quality | Locked 300-utterance paired evaluation; SI-SDR `+0.241 dB` pooled, 95% CI excludes zero | Future guidance provides a small reconstruction benefit under the matched 80k Router protocol. | Do not claim a broad perceptual improvement. |
| Future guidance has a perception–reconstruction trade-off | Pooled DNSMOS-SIG `−0.0149`, OVRL `−0.0145`; STOI `−0.0008` | The reconstruction gain is accompanied by small declines in several perceptual proxies. | Do not call DNSMOS human preference ground truth. |
| The effect depends on reverberation | Matched-file interaction: PESQ `−0.0411`, STOI `−0.00267`, SI-SDR `−0.155 dB`, OVRL `−0.0285`; CIs exclude zero | Reverberation significantly changes the Router-versus-off effect. | Do not infer a causal reverb gate or identify one mechanism. |
| t-cut is a useful fixed control | `t_cut=0.05` keeps the same direction of the split-dependent trade-off and does not remove it | A fixed early-time cutoff is a transparent engineering control, not a learned solution. | Do not present it as a validated adaptive controller. |
| Anchor V2 has residual reconstruction headroom | Clean Oracle SI-SDR `+0.448 dB` on 32 development utterances | The residual subspace contains reconstruction headroom. | Clean Oracle access is unavailable at inference; it did not improve DNSMOS-OVRL. |
| WavLM Harm Teacher is not justified by current evidence | 4-utterance N=30 oracle; OVRL gate−full `−0.000953`, CI crosses zero | The tested WavLM harm label did not expose stable perceptual headroom. | Do not generalize to every possible harm label or train a Harm Head from this result. |
| Rollout adapter repairs a proxy | Independent 4-speaker pilot: holdout clean-x0 NMSE `−54.8%` | Rollout-state prediction is a reproducible proxy bottleneck candidate. | DNSMOS-OVRL fell `0.0347`; do not call this perceptual generalization or CoF success. |
| Speaker/content auxiliaries are not currently supported | 64-speaker CV and 8-utterance ASR smoke show no stable identity/content gain | The tested auxiliary-loss routes do not justify more training under the current evidence. | Do not claim speaker or content preservation is impossible in general. |
| P7 is a bounded diagnosis | P7-A/B/C and matched-time analyses | P7 narrows candidate bottlenecks and rules out several simple explanations. | Do not claim a new diffusion SDE or a validated second innovation. |

## Recommended paper structure

1. Introduce the matched Future-guidance protocol and the motivation for examining reconstruction and perception jointly.
2. Report the locked no-reverb/with-reverb paired results and the matched-file interaction.
3. Use Anchor, Harm, Speaker/content, rollout and P7 experiments as mechanism-boundary and negative-control analyses.
4. State explicitly that the current contribution is a conditional Pareto characterization and a reproducible diagnostic protocol unless a later independent mechanism clears the evidence gates.
