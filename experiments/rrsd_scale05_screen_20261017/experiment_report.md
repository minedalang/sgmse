# Fixed RRSD scale 0.5 screen

The same 80k EMA Router was evaluated on the existing 32-file paired manifest with CUDA, PC sampling, N=30, SNR=0.5, seed 2026, and full-time RRSD (`t_max=1.0`). Only the inference multiplier changed from the Router arm: `rrsd_scale=0.5`.

| arm | PESQ | STOI | SI-SDR (dB) | DNSMOS-P808 | SIG | BAK | OVRL |
|---|---:|---:|---:|---:|---:|---:|---:|
| Router scale 1.0 | 1.749041 | 0.831925 | 7.800914 | 4.021861 | 3.477706 | 4.142772 | 3.185400 |
| scale 0.5 | 1.742957 | 0.831614 | 7.569328 | 4.033515 | 3.488255 | 4.144595 | 3.197109 |

Paired bootstrap (20,000 resamples, seed 2026) for scale 0.5 minus Router: PESQ `-0.006084` (95% CI `[-0.012363,-0.000267]`), STOI `-0.000311` (`[-0.000505,-0.000125]`), SI-SDR `-0.231586 dB` (`[-0.259808,-0.204524]`), P808 `+0.011654` (`[+0.006997,+0.016826]`), SIG `+0.010548` (`[+0.006962,+0.014689]`), BAK `+0.001823` (`[+0.000158,+0.003561]`), and OVRL `+0.011710` (`[+0.007507,+0.016296]`).

The reduced scale moves the output toward a perceptual Pareto point, but its primary reconstruction losses remain clear. It is a fixed-strength baseline, not a validated innovation or a reason for full-test expansion.
