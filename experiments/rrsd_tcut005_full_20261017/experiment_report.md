# Full-test RRSD time cutoff 0.05

This is a fixed inference-time reliability rule applied to the same 80k EMA Router checkpoint. RRSD's additional score correction is disabled only for `t < 0.05`; the base score and frontend remain unchanged. Conditions were CUDA, EMA, PC, N=30, SNR=0.5, seed 2026, and `t_max=1.0` on the complete 150-file test split.

| arm | PESQ | STOI | SI-SDR (dB) | DNSMOS-P808 | SIG | BAK | OVRL |
|---|---:|---:|---:|---:|---:|---:|---:|
| Router | 1.689556 | 0.812773 | 7.915450 | 3.976104 | 3.492479 | 4.147991 | 3.212745 |
| t_cut=0.05 | 1.687021 | 0.812577 | 7.865731 | 3.981292 | 3.496450 | 4.147720 | 3.216618 |

Paired bootstrap on the same 150 files (20,000 resamples, seed 2026), cutoff minus Router:

- PESQ `-0.002535`, 95% CI `[-0.003760,-0.001155]`;
- STOI `-0.000196`, 95% CI `[-0.000261,-0.000132]`;
- SI-SDR `-0.049719 dB`, 95% CI `[-0.053556,-0.046117]`;
- DNSMOS-P808 `+0.005188`, 95% CI `[+0.003407,+0.006968]`;
- DNSMOS-SIG `+0.003971`, 95% CI `[+0.002391,+0.005585]`;
- DNSMOS-BAK `-0.000271`, 95% CI `[-0.001909,+0.001417]`;
- DNSMOS-OVRL `+0.003873`, 95% CI `[+0.001864,+0.005916]`.

This is the strongest current Pareto candidate: it preserves nearly all Router SI-SDR while making a small but statistically stable DNSMOS improvement. It is still a hand-designed time cutoff, so it is a baseline for a future state-dependent reliability gate rather than proof of a learned Harm Teacher.
