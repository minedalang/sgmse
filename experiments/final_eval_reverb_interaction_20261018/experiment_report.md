# Matched reverb interaction analysis of the locked final evaluation (2026-10-18)

## Question

Does adding reverberation change the Router-versus-off effect on the same file IDs? This is a post-hoc analysis of the frozen final outputs, not a new evaluation protocol or tuning procedure.

## Protocol

The analysis paired each of the 150 file IDs appearing in both the `no_reverb` and `with_reverb` splits. For each metric, it computed the interaction:

`(Router - off)_with_reverb - (Router - off)_no_reverb`.

The same calculation was made for `t_cut=0.05`. Intervals use 10,000 matched-file bootstrap resamples with seed 20261018. No audio was regenerated and no clean signal was used for inference.

## Results

Router interaction relative to off:

| metric | interaction mean | 95% bootstrap CI |
|---|---:|---:|
| PESQ | -0.04110 | [-0.06326, -0.02274] |
| STOI | -0.00267 | [-0.00363, -0.00174] |
| SI-SDR | -0.15524 dB | [-0.23236, -0.07470] |
| DNSMOS-SIG | -0.02015 | [-0.03180, -0.00818] |
| DNSMOS-OVRL | -0.02854 | [-0.04191, -0.01515] |
| DNSMOS-P808 | -0.02523 | [-0.03935, -0.01142] |
| DNSMOS-BAK | +0.00924 | [-0.00344, +0.02189] |

For `t_cut=0.05`, the corresponding interactions were PESQ `-0.03330`, STOI `-0.00295`, SI-SDR `-0.22235 dB`, SIG `-0.01549`, OVRL `-0.02235`, P808 `-0.01339`, and BAK `+0.00969`; all intervals except BAK excluded zero.

## Interpretation

Reverberation systematically changes the Router effect: relative to the no-reverb condition, Router loses about 0.041 PESQ, 0.0027 STOI, 0.155 dB SI-SDR, and 0.029 DNSMOS-OVRL compared with off. This strengthens the statement that the observed conflict is an interaction with reverberant conditions, rather than a single global SNR effect.

This still does not identify the causal mechanism. Possible explanations include frontend conditioning under reverberation, phase/reconstruction behavior, or a mismatch between the Router training distribution and reverberant inputs. The analysis does not justify a new reverb gate or any test-set threshold.

## Reproduction

`analyze_reverb_interaction.py` reads the locked manifest and CSVs from `experiments/final_eval_full_20261018/`. `metrics.json` contains all metrics and intervals. No dataset or checkpoint was modified.
