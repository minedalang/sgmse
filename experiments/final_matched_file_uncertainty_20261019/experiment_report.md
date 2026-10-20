# Final matched-file uncertainty supplement: completed

One main run completed in 0.469342s (PID15768); one independent checker passed in 0.103046s (PID15823). Worker thread `01a153f1-d6c2-7f42-9c8b-323bcb2a9375` used codex_tools_compat. Both commands completed in their initial tool calls; no running exec session_id was issued. Actual tool chunk IDs: main611ee0, checker a24d72.

This is a post hoc uncertainty supplement for frozen final metrics, not a new final evaluation, independent confirmation, method selection or multiple-comparison correction. A matched file-ID keeps both no-reverb/with-reverb conditions and all three arms together. The150 matched IDs are not fabricated speaker, room or noise clusters. The two-condition equal-weight pooled estimand remains unchanged.

Fixed PCG64/default_rng seed20261019;10000 bootstrap draws of150 matched IDs;95% percentile intervals with linear quantiles. Main used index-gather averaging; the separate checker used per-draw bincount weights and separate linear endpoint bookkeeping, checking all10000×14 draw values and original/new endpoints. Shared NumPy RNG backend and lifecycle IO are acknowledged. No seed/draw/statistic tuning, automatic retry or source deletion.

All14 pooled means reproduce their original values; maximum absolute difference 5.2041704279304213e-18 (tolerance2e−10). Both historical and matched-file intervals are retained below; neither is selected by width or result.

| Contrast | Metric | Mean Δ | Historical95% CI | Matched-file95% CI | Mean absolute error |
|---|---|---:|---|---|---:|
| router_vs_off | PESQ_e | +0.0058343681 | [-0.0058247189, +0.0155295021] | [-0.0054288328, +0.0152194584] | 0 |
| router_vs_off | STOI_e | -0.0008132499 | [-0.0013439304, -0.0003085240] | [-0.0013390238, -0.0003125763] | 2.17e-19 |
| router_vs_off | SI-SDR_e | +0.2414120741 | [+0.2020647765, +0.2824502051] | [+0.2014005095, +0.2828147427] | 0 |
| router_vs_off | DNSMOS_SIG_e | -0.0148820815 | [-0.0211614514, -0.0085824798] | [-0.0209755492, -0.0087312476] | 5.2e-18 |
| router_vs_off | DNSMOS_OVRL_e | -0.0144724954 | [-0.0216559575, -0.0074032328] | [-0.0215920433, -0.0075431583] | 3.47e-18 |
| router_vs_off | DNSMOS_P808_e | -0.0103925347 | [-0.0174051530, -0.0035737307] | [-0.0172958833, -0.0036224421] | 0 |
| router_vs_off | DNSMOS_BAK_e | +0.0120406853 | [+0.0056120955, +0.0187245024] | [+0.0053098901, +0.0189920595] | 0 |
| tcut005_vs_off | PESQ_e | +0.0051673285 | [-0.0094407970, +0.0184495143] | [-0.0090278507, +0.0181086295] | 0 |
| tcut005_vs_off | STOI_e | -0.0010164930 | [-0.0014683261, -0.0005749453] | [-0.0014553465, -0.0005961070] | 2.17e-19 |
| tcut005_vs_off | SI-SDR_e | +0.1626495192 | [+0.1248873878, +0.1981885597] | [+0.1259197128, +0.1978306335] | 0 |
| tcut005_vs_off | DNSMOS_SIG_e | -0.0116540175 | [-0.0172423085, -0.0060633095] | [-0.0172408451, -0.0061591446] | 0 |
| tcut005_vs_off | DNSMOS_OVRL_e | -0.0112724031 | [-0.0178274673, -0.0050352207] | [-0.0177993253, -0.0048912018] | 1.73e-18 |
| tcut005_vs_off | DNSMOS_P808_e | -0.0050176326 | [-0.0112476349, +0.0011602493] | [-0.0113696881, +0.0011526239] | 0 |
| tcut005_vs_off | DNSMOS_BAK_e | +0.0109356882 | [+0.0047658308, +0.0173601393] | [+0.0045966620, +0.0173930674] | 0 |

Commands (each executed once):

```text
/data/coding/sgmse/env/bin/python -B /data/coding/sgmse/experiments/final_matched_file_uncertainty_20261019/bootstrap_main.py --execute-after-root-review
/data/coding/sgmse/env/bin/python -B /data/coding/sgmse/experiments/final_matched_file_uncertainty_20261019/bootstrap_check.py --validate-after-completed-run
```

Protection: all six original CSV files and original metrics JSON have identical pre/post SHA256 in both stages. All reviewed Python/config/analysis-plan hashes match pre/post. Eight prior metrics and all three user-chat export JSON files retain their preflight hashes. Primary result metrics remain byte-identical through validation and report generation. The listening exports were only hashed for preservation; their judgments were not read into or combined with these estimates. No audio/header/checkpoint/model/ASR/GPU/training/network access or WAV hashing occurred. Existing dirty repository state is retained in execution_preflight.json. Bridge39452 and listening server16022 remain alive.

Primary metrics SHA256: `cff1cb2ef8bf3919d64908d3e4f110945aef02acb8f0328dfb05265435f104a0`. Numerical records: metrics.json; independent all-draw check: independent_validation.json; fixed sampled indices/draw means: matched_bootstrap.npz. Full preservation hashes and actual PID/tool receipt: execution_receipt.json. This report was generated only from existing result JSON and preservation checks, without reanalysis.

These intervals do not supply cross-training-seed, cross-dataset, speaker/noise-cluster or speech-content evidence. The user’s12 listening cases remain a separate descriptive record and are not pooled here. No request for additional listening is required. The bounded supplement is complete and stopped.
