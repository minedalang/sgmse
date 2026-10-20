# Held-out SNR split: RRSD t_cut=0.05 vs full Router

This matched evaluation uses 151 files from the held-out `-20~-15 dB` test split. Both arms use the same 80k EMA checkpoint, CUDA PC sampler, N=30, SNR=0.5, seed 2026 and `rrsd_t_max=1.0`; the t-cut arm additionally sets `rrsd_t_cut=0.05`. The clean/noisy references are used only by the offline evaluator.

| metric | full Router | t_cut=0.05 | delta | 95% CI |
|---|---:|---:|---:|---:|
| PESQ_e | 1.776966 | 1.773410 | -0.003556 | [-0.005246, -0.002112] |
| STOI_e | 0.777289 | 0.777065 | -0.000224 | [-0.000283, -0.000166] |
| SI-SDR_e | 7.084077 | 7.022998 | -0.061079 | [-0.069718, -0.054554] |
| DNSMOS_P808_e | 3.968755 | 3.975632 | +0.006877 | [+0.004676, +0.009219] |
| DNSMOS_SIG_e | 3.414992 | 3.418158 | +0.003166 | [+0.001422, +0.004984] |
| DNSMOS_BAK_e | 4.133473 | 4.132861 | -0.000612 | [-0.001742, +0.000613] |
| DNSMOS_OVRL_e | 3.112830 | 3.116051 | +0.003221 | [+0.001271, +0.005334] |

Bootstrap: paired resampling, 20,000 replicates, seed 2026. Positive counts are the number of per-utterance deltas above zero.

Interpretation: this split tests generalization of the hand-designed late-step cutoff. A DNSMOS gain with a small reconstruction loss would support it as a Pareto baseline; it does not validate the WavLM Harm Teacher or a learned gate.
