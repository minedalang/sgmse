# rrsd_future_gate_1000

## Design

This is a 1,000-step short validation of the explicit `future_grad` RRSD gate. It starts from the 80k EMA Router, freezes the base model, and trains the Router/gate on the CV split. The inference comparison uses the existing 32-file repeated-speaker CV manifest, CUDA, EMA, RRSD PC sampler, N=30, SNR=0.5, and seed 2026. Dataset files were not modified.

## Training status

The 1,000 steps completed with `train_loss_epoch=403.185`. The primary checkpoint `epoch=0-step=1000.ckpt` was independently loaded with `global_step=1000` and 548 state-dict entries. Lightning's additional `last.ckpt` save failed afterward with `OSError: [Errno 28] No space left on device`; this does not invalidate the primary checkpoint.

## Paired SE metrics on 32 utterances

The exact per-utterance values are in `paired_metrics.csv`. Relative to the existing 80k Router output:

- PESQ: `-0.005117`, bootstrap 95% CI `[-0.011137, +0.000519]`.
- STOI: `-0.000591`, bootstrap 95% CI `[-0.001009, -0.000220]`.
- SI-SDR: `-0.283873 dB`, bootstrap 95% CI `[-0.326961, -0.242262]`.

Against Router-off, the gated model remains better in PESQ (`+0.007905`), STOI (`+0.000633`), and SI-SDR (`+0.199948 dB`), but it loses part of the original Router gain. This short run therefore does not support expanding the future-grad gate without a stronger calibration or training hypothesis.

DNSMOS values and paired bootstrap intervals are recorded in `metrics.json` and `dnsmos_paired_summary.json`. CAM++ was not recomputed because this A experiment is a routing-quality test; the existing independent speaker probe is tracked separately.

## DNSMOS on the same 32 utterances

The per-file DNSMOS outputs for Off, Router, and future-grad gate are stored in `off_dnsmos_metrics.csv`, `router_dnsmos_metrics.csv`, and `dnsmos_metrics.csv`; paired bootstrap results are in `dnsmos_paired_summary.json`.

| metric | Off | Router | future-grad gate | gate − Router (95% CI) | gate − Off (95% CI) |
|---|---:|---:|---:|---:|---:|
| P808 | 3.94307 | 3.91332 | 3.93230 | +0.01898 `[+0.01070,+0.02851]` | -0.01078 `[-0.01588,-0.00603]` |
| SIG | 3.44632 | 3.41879 | 3.43922 | +0.02043 `[+0.01198,+0.02964]` | -0.00710 `[-0.01103,-0.00339]` |
| BAK | 4.11865 | 4.11704 | 4.11934 | +0.00230 `[-0.00306,+0.00938]` | +0.00069 `[-0.00124,+0.00258]` |
| OVRL | 3.13110 | 3.10088 | 3.12364 | +0.02275 `[+0.01226,+0.03408]` | -0.00747 `[-0.01216,-0.00317]` |

The gate recovers part of the Router's DNSMOS loss, especially SIG and OVRL, but does not exceed Router-off on P808, SIG, or OVRL. Combined with the reconstruction metrics, this is a Pareto trade-off rather than a validated overall improvement. The current evidence is sufficient to stop this exact 1,000-step gate variant and avoid a longer run without changing the hypothesis or gate objective.
