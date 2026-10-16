# Soft late-step RRSD cutoff screen (32 utterances)

This screen keeps the same 80k EMA Router and disables only the additional RRSD correction below `t=0.05` by a partial factor. `t_cut_scale=0` is the existing hard cutoff; 1.0 would be the full Router at that step. Conditions are CUDA, EMA, PC, N=30, SNR=0.5, seed 2026.

## Means

| arm | PESQ | STOI | SI-SDR (dB) |
|---|---:|---:|---:|
| router | 1.749041 | 0.831925 | 7.801027 |
| hard_cut005 | 1.748753 | 0.831805 | 7.743281 |
| scale025 | 1.749039 | 0.831832 | 7.757927 |
| scale05 | 1.748380 | 0.831871 | 7.772450 |
| scale075 | 1.749108 | 0.831898 | 7.786787 |

## Paired deltas versus full Router

| arm | metric | delta | 95% CI | positive |
|---|---|---:|---:|---:|
| scale025 | PESQ | -0.000002 | [-0.003351, +0.003493] | 12/32 |
| scale025 | STOI | -0.000093 | [-0.000139, -0.000046] | 6/32 |
| scale025 | SI-SDR | -0.043100 | [-0.048595, -0.037820] | 0/32 |
| scale05 | PESQ | -0.000661 | [-0.006078, +0.005941] | 11/32 |
| scale05 | STOI | -0.000054 | [-0.000086, -0.000021] | 6/32 |
| scale05 | SI-SDR | -0.028576 | [-0.032220, -0.025080] | 0/32 |
| scale075 | PESQ | +0.000067 | [-0.001137, +0.001291] | 13/32 |
| scale075 | STOI | -0.000027 | [-0.000049, -0.000003] | 7/32 |
| scale075 | SI-SDR | -0.014240 | [-0.016051, -0.012510] | 0/32 |

The existing hard-cut DNSMOS values are in `experiments/rrsd_tcut_screen_dns_20261017/`; this screen initially evaluates reconstruction metrics. DNSMOS is added only for the surviving soft setting(s), because it is the slow evaluator.

## DNSMOS screen

| arm | P808 | SIG | BAK | OVRL |
|---|---:|---:|---:|---:|
| router | 4.021861 | 3.477706 | 4.142772 | 3.185399 |
| hard_cut005 | 4.026481 | 3.480285 | 4.143838 | 3.188660 |
| scale025 | 4.026206 | 3.478284 | 4.143713 | 3.186610 |
| scale05 | 4.024351 | 3.478298 | 4.142631 | 3.186172 |
| scale075 | 4.023069 | 3.478038 | 4.142504 | 3.185641 |

### Soft arm DNSMOS deltas versus Router

| arm | metric | delta | 95% CI | positive |
|---|---|---:|---:|---:|
| scale025 | DNSMOS_P808_e | +0.004345 | [+0.001706, +0.007339] | 24/32 |
| scale025 | DNSMOS_SIG_e | +0.000578 | [-0.001009, +0.002253] | 17/32 |
| scale025 | DNSMOS_BAK_e | +0.000941 | [-0.000623, +0.003183] | 19/32 |
| scale025 | DNSMOS_OVRL_e | +0.001211 | [-0.000681, +0.003199] | 17/32 |
| scale05 | DNSMOS_P808_e | +0.002490 | [+0.000242, +0.004909] | 20/32 |
| scale05 | DNSMOS_SIG_e | +0.000592 | [-0.000712, +0.002123] | 16/32 |
| scale05 | DNSMOS_BAK_e | -0.000141 | [-0.001386, +0.001287] | 15/32 |
| scale05 | DNSMOS_OVRL_e | +0.000773 | [-0.000940, +0.002673] | 16/32 |
| scale075 | DNSMOS_P808_e | +0.001207 | [-0.000416, +0.003032] | 21/32 |
| scale075 | DNSMOS_SIG_e | +0.000331 | [-0.000943, +0.001617] | 17/32 |
| scale075 | DNSMOS_BAK_e | -0.000268 | [-0.001447, +0.000788] | 13/32 |
| scale075 | DNSMOS_OVRL_e | +0.000241 | [-0.001288, +0.001724] | 16/32 |

## Interpretation and exit decision

The partial gate forms a reconstruction-quality continuum. Relative to the full Router, SI-SDR losses were `-0.043100`, `-0.028576`, and `-0.014240 dB` for scales 0.25, 0.5, and 0.75. The corresponding DNSMOS-OVRL changes were `+0.001211`, `+0.000773`, and `+0.000241`, with all three 95% CIs crossing zero. Thus soft attenuation reduces the hard cutoff's SI-SDR cost but does not reproduce its reliable OVRL gain on this screen. Keep `scale=0.25` as a reconstruction-friendly Pareto ablation; do not expand soft variants to the full test set or train a new model based on this screen alone. The hard cutoff remains the stronger perceptual baseline, while the soft gate is a useful controllable trade-off rather than a validated learned innovation.
