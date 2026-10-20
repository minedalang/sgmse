# Matched full Router re-evaluation

The 80k EMA Router checkpoint was re-evaluated on the complete 150-file test split using the current fixed inference code (`t_max=1.0`), CUDA, PC sampling, N=30, SNR=0.5, seed 2026. The output is a new directory and does not overwrite the older unpaired result.

| metric | Router 80k matched |
|---|---:|
| PESQ | 1.689556 ± 0.613 |
| STOI | 0.812773 ± 0.1540 |
| SI-SDR (dB) | 7.915450 ± 5.902 |
| DNSMOS-P808 | 3.976104 ± 0.219 |
| DNSMOS-SIG | 3.492479 ± 0.195 |
| DNSMOS-BAK | 4.147991 ± 0.077 |
| DNSMOS-OVRL | 3.212745 ± 0.240 |

Compared with the matched 1,000-step `conflict_weight=0.001` arm, the conflict arm changes OVRL by `+0.001870` (95% CI `[+0.000031,+0.003691]`) and SIG by `+0.001993` (`[+0.000351,+0.003607]`), but lowers SI-SDR by `0.046902 dB` (`[-0.050570,-0.043337]`). A same-budget `conflict_weight=0` control is required for causal attribution; that control is recorded separately.
