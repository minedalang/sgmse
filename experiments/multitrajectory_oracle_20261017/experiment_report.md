# Multi-trajectory stochastic Router Oracle (16 files × 4 seeds)

## Setup

The 80k EMA Router checkpoint was sampled with the unchanged Router on the first 16 manifest files. Each file used seeds 2026, 2027, 2028, and 2029, PC sampler `N=30`, `snr=0.5`, CUDA, and identical model settings. Enhanced wavs were written only under `/dev/shm/multitraj_router_wavs_16`.

Each file required four full trajectories, with `30` reported NFE per trajectory: `120` score-network evaluations per input, approximately four times a single `N=30` trajectory, before selector/evaluator overhead.

For each file, a clean-reference diagnostic Oracle selected the candidate with maximum DNSMOS OVRL subject to SI-SDR being at least the seed-2026 baseline minus 0.05 dB. This clean reference is used only for offline selection and is not a deployable controller.

## Means

| selection | PESQ | STOI | SI-SDR (dB) | P808 | SIG | BAK | OVRL |
|---|---:|---:|---:|---:|---:|---:|---:|
| Seed 2026 baseline | 1.844401 | 0.814506 | 8.145444 | 3.960789 | 3.422922 | 4.104707 | 3.085323 |
| Clean-reference Oracle | 1.828219 | 0.814663 | 8.264103 | 3.940900 | 3.460351 | 4.125275 | 3.135063 |

## Paired bootstrap Oracle minus seed 2026 (20,000 resamples, 95% CI)

| metric | delta | CI |
|---|---:|---:|
| PESQ | -0.016182 | [-0.052356, +0.019098] |
| STOI | +0.000156 | [-0.005028, +0.004854] |
| SI-SDR (dB) | +0.118659 | [+0.053095, +0.190207] |
| DNSMOS P808 | -0.019889 | [-0.070962, +0.025531] |
| DNSMOS SIG | +0.037429 | [+0.018343, +0.058219] |
| DNSMOS BAK | +0.020569 | [+0.002562, +0.044737] |
| DNSMOS OVRL | +0.049741 | [+0.025177, +0.077137] |

## Decision

Stochastic Router trajectories contain measurable clean-reference perceptual headroom: OVRL improves by +0.049741 with a CI strictly above zero while SI-SDR also improves by +0.118659 dB. This supports a follow-up deployable trajectory selector/controller study. The Oracle remains diagnostic only; no training or production-path changes were made.

Per-trajectory and per-file Oracle records are in `metrics.json`.

## No-reference selector probe

Selecting the candidate with the highest DNSMOS-OVRL uses only the enhanced waveforms and no clean/reference signal. On these 16 development files it gave OVRL delta `+0.054821` (95% CI `[+0.027659,+0.085534]`), SI-SDR delta `+0.094852 dB` (95% CI `[+0.004535,+0.180813]`), SIG delta `+0.041294`, and PESQ delta `-0.013671`. It selected the same seed as the clean constrained Oracle for 14/16 files. The worst per-file SI-SDR change was `-0.331909 dB`, so a deployment method needs a risk safeguard. This is an in-split diagnostic; held-out evaluation on the remaining 16 files is required.
