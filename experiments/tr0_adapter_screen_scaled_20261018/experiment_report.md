# TR-0 scaled trust-region screen — 2026-10-18

The first TR-0 screen used trust weight `1.0` and produced indistinguishable `score_only` and `trust_region` trajectories because the raw trust penalty was around `1e-5` while score loss ranged from roughly `1e1` to `1e3`. That configuration is retained as a loss-scale failure, not as evidence against the mechanism.

This follow-up keeps the same data, seeds, arms, adapter, optimizer, and 200-step budget, changing only the trust weight to `1e4`. The scale is based on the observed training-loss units from the first screen and is not selected from validation or test outputs. The script records the full per-step history and saves adapter state only.

This remains a training proxy screen. It requires fresh-CV waveform inference and independent DNSMOS/PESQ/STOI/SI-SDR evaluation before any scientific conclusion.

## Follow-up fresh-CV proxy decision

The fresh 32-speaker × 2-utterance single-step proxy was then run with identical `t` and noise for baseline and all four adapter states. All adapters worsened clean-reference `x0` MSE:

| Arm | Mean delta vs baseline | Speaker-cluster 95% CI |
|---|---:|---:|
| score-only, seed 20261018 | `+2.215e-6` | `[-0.090e-6, +5.041e-6]` |
| trust-region, seed 20261018 | `+2.118e-6` | `[-0.164e-6, +4.881e-6]` |
| score-only, seed 20261019 | `+4.212e-6` | `[+2.714e-6, +5.881e-6]` |
| trust-region, seed 20261019 | `+6.178e-6` | `[+4.014e-6, +8.624e-6]` |

This triggers the proxy stop condition. No full PC waveform or DNSMOS evaluation was run, and no adapter screen expansion is justified by this result.
