# Held-out multi-trajectory Router selector (16 files × 4 seeds)

The 16 held-out files (manifest rows 16–31) were evaluated with the unchanged 80k EMA Router, PC `N=30`, `snr=0.5`, and seeds 2026–2029. Wavs are temporary under `/dev/shm/multitraj_router_wavs_heldout16`.

This selector evaluates four complete trajectories per input (`30` NFE each), or `120` score-network evaluations per input—approximately four times the compute of one `N=30` trajectory, before DNSMOS selection overhead.

The fixed selector uses no clean reference: it chooses the candidate with highest DNSMOS OVRL among the four trajectories. The clean constrained Oracle is reported only as an upper-bound diagnostic, selecting maximum OVRL subject to SI-SDR >= seed-2026 baseline - 0.05 dB. Paired bootstrap uses 20,000 resamples, seed 2026.

| selection | PESQ | STOI | SI-SDR | P808 | SIG | BAK | OVRL |
|---|---:|---:|---:|---:|---:|---:|---:|
| Seed 2026 | 1.564771 | 0.775334 | 5.724925 | 3.887527 | 3.386773 | 4.102021 | 3.072068 |
| Fixed no-reference selector | 1.576163 | 0.776937 | 5.749771 | 3.905637 | 3.434977 | 4.142043 | 3.136623 |
| Clean constrained Oracle | 1.571002 | 0.778017 | 5.776384 | 3.896807 | 3.426058 | 4.133837 | 3.123248 |

## Paired bootstrap delta versus seed 2026 (95% CI)

| metric | Fixed selector | Clean Oracle |
|---|---:|---:|
| PESQ | +0.011392 [-0.006129, +0.028636] | +0.006230 [-0.009006, +0.020939] |
| STOI | +0.001603 [-0.002023, +0.005064] | +0.002683 [+0.000236, +0.005371] |
| SI-SDR (dB) | +0.024846 [-0.027057, +0.074982] | +0.051459 [+0.018165, +0.087513] |
| DNSMOS P808 | +0.018109 [-0.010213, +0.049035] | +0.009280 [-0.015779, +0.036260] |
| DNSMOS SIG | +0.048204 [+0.026813, +0.071382] | +0.039285 [+0.017160, +0.064101] |
| DNSMOS BAK | +0.040022 [+0.015419, +0.067501] | +0.031817 [+0.008376, +0.059367] |
| DNSMOS OVRL | +0.064555 [+0.039985, +0.090509] | +0.051181 [+0.025002, +0.079162] |

The fixed no-reference selector improves OVRL by +0.064555 with a CI strictly above zero on held-out files, while also improving SIG and BAK. PESQ, STOI, and SI-SDR intervals include zero. This validates stochastic trajectory selection without using clean references; no training, data, checkpoint, or production inference path was changed.

## Exploratory no-reference risk guard

We tested a fixed fallback guard using the seed-2026 waveform as baseline. A DNSMOS-OVRL-selected candidate was accepted only when its normalized log-STFT distance from that baseline was below a threshold learned from the development half. This guard uses no clean signal and is not a trained model.

The development median threshold (`0.1427`) accepted 5/16 held-out candidates, retained OVRL delta `+0.0033`, and removed all negative SI-SDR changes on the held-out half. Looser thresholds retained larger OVRL gains (`+0.0333` to `+0.0536`) but still allowed negative SI-SDR cases, including the `-0.154 dB` worst case. This simple distance guard therefore does not meet the desired Pareto criterion and is recorded as a diagnostic negative result.

## Trajectory-consistency diagnostic

We also computed a clean-free consistency signal for each trajectory: the mean normalized log-STFT distance to the other three seeds for the same noisy input. This signal is available at inference time and does not use clean audio.

On the development trajectories, consistency correlated with DNSMOS-OVRL at `-0.747` and with SI-SDR at `-0.952`. On held-out trajectories, the correlations were `-0.634` and `-0.820`, respectively. Lower inter-trajectory distance tends to identify higher-quality and safer candidates, but it is not a sufficient selector.

Thresholding the signal using development quantiles confirmed the trade-off: the conservative 25th-percentile threshold produced only `+0.0033` OVRL with no negative SI-SDR case on held-out files; the 70th-percentile threshold produced `+0.0416` OVRL but four negative SI-SDR file changes, with a worst case of `-0.154 dB`. The consistency signal is useful for risk ranking, but by itself does not satisfy the perception/reconstruction Pareto requirement. No learned verifier or production-path change was started.
