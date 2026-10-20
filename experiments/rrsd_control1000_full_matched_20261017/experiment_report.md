# Matched 1000-step control for conflict-aware loss

This control was initialized from the same 80k EMA Router checkpoint as the `conflict_weight=0.001` run and trained for the same 1,000 steps, batch size, learning rate, seed, and data path, with `conflict_weight=0`. It was evaluated on all 150 test files with CUDA EMA RRSD PC sampling, N=30, SNR=0.5, seed 2026, and `rrsd_t_max=1.0`. No dataset or existing checkpoint was modified.

| arm | PESQ | STOI | SI-SDR (dB) | P808 | SIG | BAK | OVRL |
|---|---:|---:|---:|---:|---:|---:|---:|
| control 1000 | 1.688873 | 0.812717 | 7.891076 | 3.977340 | 3.494130 | 4.147892 | 3.214205 |
| conflict 0.001 1000 | 1.688806 | 0.812657 | 7.868548 | 3.979798 | 3.494471 | 4.147327 | 3.214615 |

Paired bootstrap used the same 150-file manifest, seed 2026, and 20,000 resamples. Conflict minus control was PESQ `-0.000067` (95% CI `[-0.002459,+0.002170]`), STOI `-0.000061` (`[-0.000094,-0.000027]`), SI-SDR `-0.022527 dB` (`[-0.024848,-0.020325]`), P808 `+0.002458` (`[+0.001406,+0.003565]`), SIG `+0.000341` (`[-0.000837,+0.001524]`), BAK `-0.000565` (`[-0.001484,+0.000321]`), and OVRL `+0.000410` (`[-0.000909,+0.001714]`).

The conflict loss therefore does not provide a reliable OVRL or SIG improvement beyond ordinary 1,000-step adaptation. Its only clear DNSMOS change is a small P808 increase, while SI-SDR and STOI decline. The current conflict-aware route should not be expanded to a long training run without a new hypothesis.
