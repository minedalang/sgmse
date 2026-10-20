# n30 full versus hard-gate paired metrics

Evaluated the four outputs for fileids 2673, 2972, 7247, and 4431 against their corresponding `clean_fileid_*.wav` references at 16 kHz. Metrics are PESQ wideband, STOI, and SI-SDR; `metrics.json` and `paired_metrics.csv` contain per-file values.

The paired difference is gate minus full. Bootstrap uses 10,000 paired resamples with seed 2026:

- PESQ: +0.000158, 95% CI [-0.000006, +0.000388]
- STOI: +0.000007, 95% CI [-0.000005, +0.000025]
- SI-SDR: -0.000018 dB, 95% CI [-0.000053, +0.000011]

DNSMOS was intentionally not run.
