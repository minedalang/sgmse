# DNSMOS read-only comparison: harm-oracle N30-4

This is a read-only evaluation of four existing WAV pairs from `rrsd_harm_oracle_n30_4_full` and `rrsd_harm_oracle_n30_4_gate`. No inference, training, dataset, or checkpoint changes were made.

DNSMOS was computed with the mymetrics `calc_dnsmos` evaluator. ONNX Runtime warned that no CUDA plugin device was registered; the resolver reported `cuda:0`, but actual CUDA execution was not verified and CPU fallback may have occurred.

| metric | full | gate | gate − full | bootstrap 95% CI |
|---|---:|---:|---:|---:|
| P808 | 4.05849 | 4.05772 | −0.00077 | [−0.00215, +0.00137] |
| SIG | 3.51625 | 3.51562 | −0.00063 | [−0.00200, +0.00048] |
| BAK | 4.17182 | 4.17092 | −0.00090 | [−0.00236, +0.000002] |
| OVRL | 3.23425 | 3.23330 | −0.00095 | [−0.00291, +0.00032] |

Bootstrap uses 10,000 resamples of the four utterances and is exploratory only; it is not a speaker-level confidence interval or a generalization claim. Per-file values and hashes are in `paired_dnsmos.csv`; complete output is in `run.log` and `metrics.json`.
