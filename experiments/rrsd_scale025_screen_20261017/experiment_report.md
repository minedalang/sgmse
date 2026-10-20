# Fixed RRSD scale 0.25 screen

The same 80k EMA Router was evaluated on the existing 32-file paired manifest with CUDA, PC sampling, N=30, SNR=0.5, seed 2026. Only `rrsd_scale=0.25` changed from the Router arm.

| arm | PESQ | STOI | SI-SDR (dB) | DNSMOS-P808 | SIG | BAK | OVRL |
|---|---:|---:|---:|---:|---:|---:|---:|
| Router scale 1.0 | 1.749041 | 0.831925 | 7.800914 | 4.021861 | 3.477706 | 4.142772 | 3.185400 |
| scale 0.25 | 1.741636 | 0.831304 | 7.431466 | 4.038617 | 3.493689 | 4.144602 | 3.202645 |

Paired bootstrap (20,000 resamples, seed 2026) for scale 0.25 minus Router: PESQ `-0.007404` (95% CI `[-0.015403,-0.000842]`), STOI `-0.000622` (`[-0.000908,-0.000342]`), SI-SDR `-0.369448 dB` (`[-0.413645,-0.327515]`), P808 `+0.016756` (`[+0.010668,+0.023642]`), SIG `+0.015982` (`[+0.010662,+0.021937]`), BAK `+0.001830` (`[-0.000599,+0.004286]`), and OVRL `+0.017246` (`[+0.010957,+0.024069]`).

Scale 0.25 does not improve the trade-off over scale 0.5: its DNSMOS gain is larger but its SI-SDR/PESQ/STOI losses are also larger. It is retained as a Pareto reference only.
