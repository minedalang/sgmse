# Full 150-file conflict-weight 0.001 validation

The 1000-step conflict-weight 0.001 checkpoint was evaluated on all 150 DNS test files with PC sampling, N=30, snr=0.5, seed 2026, EMA, RRSD scale 1.0, and explicit `rrsd_t_max=1.0`.

| metric | weight 0.001 full |
|---|---:|
| PESQ | 1.688806 |
| STOI | 0.812657 |
| SI-SDR (dB) | 7.868548 |
| DNSMOS P808 | 3.979798 |
| DNSMOS SIG | 3.494471 |
| DNSMOS BAK | 4.147327 |
| DNSMOS OVRL | 3.214615 |

Compared with the complete selective full test on the same files, paired deltas are PESQ +0.015780 (95% CI [+0.012049,+0.020223]), STOI +0.000942 ([+0.000743,+0.001145]), SI-SDR +0.240226 dB ([+0.223095,+0.258043]), P808 -0.022046 ([-0.027946,-0.016167]), SIG -0.017865 ([-0.022741,-0.013375]), BAK +0.002109 ([-0.001129,+0.005079]), and OVRL -0.017001 ([-0.022607,-0.011897]).

The project’s reported 5ul7u03z 300k baseline aggregate is P808 4.000, SIG 3.531, BAK 4.148, OVRL 3.248, PESQ 1.683, STOI 0.8140, and SI-SDR 7.376 dB. Relative differences are stored in `metrics.json`; this checkpoint improves PESQ and SI-SDR while reducing STOI and DNSMOS perceptual scores.

Artifacts: `experiment_logs/rrsd_conflict_weight001_full_test/infer/`, `primary_metrics.csv`, and `metrics/metrics.csv`. DNSMOS used the mymetrics CPU evaluator; the known CUDA EP incompatibility was not used.

## Matched 1000-step control comparison (2026-10-17)

A `conflict_weight=0` control initialized from the same 80k Router and trained for the same 1,000 steps was evaluated on the same 150 files. The full paired comparison is in `matched_comparison.json`. Relative to this control, `conflict_weight=0.001` changed PESQ by `-0.000067` (95% CI `[-0.002459,+0.002170]`), STOI by `-0.000061` (`[-0.000094,-0.000027]`), SI-SDR by `-0.022527 dB` (`[-0.024848,-0.020325]`), P808 by `+0.002458` (`[+0.001406,+0.003565]`), SIG by `+0.000341` (`[-0.000837,+0.001524]`), BAK by `-0.000565` (`[-0.001484,+0.000321]`), and OVRL by `+0.000410` (`[-0.000909,+0.001714]`). The apparent OVRL gain versus the 80k Router is therefore mostly attributable to the extra 1,000-step adaptation, not the conflict term itself.
