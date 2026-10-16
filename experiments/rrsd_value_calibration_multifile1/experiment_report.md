# RRSD value calibration multifile pilot 1

Completed successfully on 4 files with 8 steps each (N=8, K=4), using the 80k EMA checkpoint. Total records: 32.

Files: 000_fileid_2673.wav, 001_fileid_2972.wav, 002_fileid_7247.wav, 003_fileid_4431.wav.

## Per-step aggregate results

| step | alpha mean (mean / median) | alpha max mean | best candidate counts | mean relative gain by candidate (0, .25, .5, .75, 1) |
|---:|---:|---:|---|---|
| 0 | 0.000000 / 0.000000 | 0.000000 | {'0.0': 4} | 0.000000, 0.000000, 0.000000, 0.000000, 0.000000 |
| 1 | 0.000547 / 0.000545 | 0.022942 | {'0.0': 4} | 0.000000, -0.000124, -0.000219, -0.000341, -0.000449 |
| 2 | 0.007841 / 0.007860 | 0.058334 | {'1.0': 4} | 0.000000, 0.001336, 0.002569, 0.003774, 0.004929 |
| 3 | 0.024209 / 0.024439 | 0.073619 | {'1.0': 4} | 0.000000, 0.005499, 0.010855, 0.016130, 0.021278 |
| 4 | 0.045726 / 0.046058 | 0.082884 | {'1.0': 4} | 0.000000, 0.017590, 0.034981, 0.052120, 0.069039 |
| 5 | 0.058392 / 0.058503 | 0.082165 | {'1.0': 4} | 0.000000, 0.026761, 0.052935, 0.078499, 0.103483 |
| 6 | 0.063045 / 0.063433 | 0.079297 | {'1.0': 4} | 0.000000, 0.032407, 0.063807, 0.094227, 0.123688 |
| 7 | 0.025225 / 0.026000 | 0.038757 | {'0.0': 4} | 0.000000, 0.000000, 0.000000, 0.000000, 0.000000 |

## Per-file best-candidate sequences

- 000_fileid_2673.wav: 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0
- 001_fileid_2972.wav: 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0
- 002_fileid_7247.wav: 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0
- 003_fileid_4431.wav: 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0

Exit code: 0; wall time: 219.928 seconds. GPU before and after: NVIDIA L40S, 0 MiB / 49140 MiB, 0% utilization.

Limitations: one deterministic run per file, so no confidence intervals are estimated. The reported gain is the calibration proxy from the runner; DNSMOS was not run. No code, dataset, or checkpoint was modified; no training was run.
