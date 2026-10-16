# RRSD value calibration multifile 2

Completed successfully on 8 utterances with 8 states each (N=8, K=4), using the 80k EMA checkpoint. Total records: 64.

## Aggregate results

| step | mean alpha | best candidate counts | mean relative gain at scale=1.0 | bootstrap 95% CI |
|---:|---:|---|---:|---:|
| 0 | 0.000000 | {'0.0': 8} | 0.000000 | [0.000000, 0.000000] |
| 1 | 0.000525 | {'0.0': 6, '0.25': 1, '1.0': 1} | -0.000273 | [-0.000471, -0.000016] |
| 2 | 0.007762 | {'1.0': 8} | 0.004446 | [0.003010, 0.005969] |
| 3 | 0.024257 | {'1.0': 7, '0.0': 1} | 0.025456 | [0.011872, 0.039452] |
| 4 | 0.045881 | {'1.0': 8} | 0.066069 | [0.045941, 0.084824] |
| 5 | 0.058496 | {'1.0': 8} | 0.095821 | [0.086575, 0.105483] |
| 6 | 0.062078 | {'1.0': 8} | 0.114777 | [0.107133, 0.121986] |
| 7 | 0.024231 | {'0.0': 8} | 0.000000 | [0.000000, 0.000000] |

## Interpretation

All eight utterances share the same best-candidate schedule: scale 0 at steps 0–1 and 7, scale 1 at steps 2–6. The proxy gain is monotonic over candidate scale in steps 2–6 and peaks at step 6. This supports a selective time-region routing hypothesis.

## Limits

The result is an offline deterministic proxy calibration. It uses clean targets to score future x0 estimates and does not establish final waveform improvement. No confidence interval over seeds or sampling noise is available; bootstrap intervals above resample utterances only. No training, enhanced audio, DNSMOS, PESQ, SI-SDR, or STOI evaluation was run. No dataset or checkpoint was modified.
