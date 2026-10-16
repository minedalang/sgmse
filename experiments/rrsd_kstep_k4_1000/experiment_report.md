# RRSD K-step k4, 1000-step evaluation

Evaluation completed on 32 utterances from `experiment_logs/rrsd_kstep_k4_eval/k4_w0_25_1000`.

## Reproduction

Command (run from an isolated temporary directory so the repository-root historical `metrics_by_utterance.csv` was not overwritten):

```text
/data/coding/mymetrics/env/bin/python /data/coding/mymetrics/get_metrics_for_papertable_forDNS.py --clean_dir experiment_logs/rrsd_tcut_screen/clean --noisy_dir experiment_logs/rrsd_tcut_screen/noisy --enhanced_dir experiment_logs/rrsd_kstep_k4_eval/k4_w0_25_1000 --dnsmos --dnsmos_device cpu
```

Exit code: **0**. The generated file was copied to `experiments/rrsd_kstep_k4_1000/metrics_by_utterance.csv`; the pre-existing root file was preserved.

## Corpus metrics (mean ± SD)

| Metric | Enhanced |
|---|---:|
| PESQ | 1.7466 ± 0.5792 |
| STOI | 0.8314 ± 0.1068 |
| SI-SDR (dB) | 7.6211 ± 4.9005 |
| DNSMOS P808 | 4.0301 ± 0.1291 |
| DNSMOS OVRL | 3.1909 ± 0.1770 |
| segSNR (dB) | 8.3693 ± 3.6945 |
| WSS | 37.1526 ± 12.3548 |

Relative to noisy, mean improvements were ΔPESQ **+0.5413**, ΔSTOI **+0.1700**, and ΔSI-SDR **+19.7857 dB**.

## Paired comparison with existing router results

The existing 32-utterance `paired_metrics.csv` was joined by filename. Differences below are K4 minus router; 95% intervals are percentile bootstrap intervals over paired utterances.

| Metric | Mean difference | 95% bootstrap CI | Positive / negative |
|---|---:|---:|---:|
| PESQ | -0.00243 | [-0.00847, +0.00282] | 15 / 17 |
| STOI | -0.000563 | [-0.000854, -0.000310] | 7 / 25 |
| SI-SDR (dB) | -0.17996 | [-0.19520, -0.16407] | 0 / 32 |

The full per-utterance table and machine-readable summary are stored alongside this report. No checkpoints or source data were deleted or overwritten.

## Execution notes

The evaluator took approximately 10.5 minutes on CPU DNSMOS. No evaluator failure occurred; exit code was 0.
