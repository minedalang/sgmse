# RRSD selective full test

150 noisy test files were enhanced with the 80k EMA checkpoint using PC sampling, N=30, snr=0.5, seed 2026, and RRSD active only for 0.15 <= t <= 0.75. Inference completed successfully (11:18:41–11:26:42 UTC, exit 0).

Primary reference metrics over all 150 files:

| metric | selective mean |
|---|---:|
| PESQ WB | 1.673025 |
| STOI | 0.811715 |
| SI-SDR (dB) | 7.628322 |

Per-file primary metrics are in `experiment_logs/rrsd_selective_full_test/metrics/primary_metrics.csv`. DNSMOS evaluation was launched through `/data/coding/mymetrics/env`; CUDA EP was unavailable and torchmetrics fell back to CPU. Its output is recorded in `experiment_logs/rrsd_selective_full_test/metrics/selective_metrics.csv`.

The previously found 150-file enhanced directories have no traceable generation command/checkpoint metadata, so they are not used as off or full Router controls. No paired control comparison or paired bootstrap interval is claimed for this run.

DNSMOS evaluation completed over all 150 files (the requested CUDA mode was unavailable in the installed ONNX runtime and automatically fell back to CPU): P808 4.002, SIG 3.512, BAK 4.145, OVRL 3.232. Full per-file output is `experiment_logs/rrsd_selective_full_test/metrics/selective_metrics.csv`; `metrics.json` includes standard deviations and bootstrap 95% CIs for each mean.
