# Fixed dev-crop additivity diagnosis: completed

One CPU-only execution completed in 0.009295s, PID38411, exit0 (tool chunk b2fe85; completed in initial call, no running session_id). No model, checkpoint, other WAV, network or training access. Reproducible command:

```text
/data/coding/sgmse/env/bin/python -B /data/coding/sgmse/experiments/p8_cpel_mp_pilot_20261019/additivity_diagnosis/diagnose.py
```

The exact frozen dev_registry[0] clean/noisy/noise sources were used with center start223680 and32640 samples, zero lag and no gain fit/reselection. Raw float64 crop arithmetic uses the same unmodified common gain relationship: a common scale cancels in the relative RMS. No source-specific normalization was applied.

| Source | Container | Subtype | Bits | Quantum | Negative endpoint count | Positive endpoint count |
|---|---|---|---:|---:|---:|---:|
| X | WAV | PCM_16 | 16 | 3.0517578125e-05 | 0 | 0 |
| S | WAV | PCM_16 | 16 | 3.0517578125e-05 | 0 | 0 |
| N | WAV | PCM_16 | 16 | 3.0517578125e-05 | 0 | 0 |

Endpoint counts cover the selected crop only. Zero endpoint counts show no code-endpoint saturation there; they do not prove that no earlier processing/clipping occurred.

Raw residual X−S−N RMS=2.14332449839627e-05; maximum absolute residual=3.0517578125e-05. Original X RMS=0.00638732894610562. Relative RMS=0.00335558809712323, reproducing attempt03's float32 value 0.00335558806546032 to about3.17×10⁻¹¹ absolute difference. It remains above the unchanged0.001 gate.

| Quantile | Signed residual | Absolute residual |
|---|---:|---:|
| 0 | 0 | 0 |
| 0.01 | 0 | 0 |
| 0.05 | 0 | 0 |
| 0.25 | 0 | 0 |
| 0.5 | 0 | 0 |
| 0.75 | 3.0517578125e-05 | 3.0517578125e-05 |
| 0.95 | 3.0517578125e-05 | 3.0517578125e-05 |
| 0.99 | 3.0517578125e-05 | 3.0517578125e-05 |
| 1 | 3.0517578125e-05 | 3.0517578125e-05 |

All three sources are PCM16, q=2⁻¹⁵. Conditional worst-case final nearest-rounding bound0.5(qX+qS+qN)=4.57763671875e-05. With numeric tolerance1e−12, 0/32640 samples exceed that bound. The observed maximum is one quantum; the signed residual quantiles are0 or+q.

The residual is compatible with the conditional final-rounding bound and does not contradict that narrow explanation. This does not prove exact original synthesis, noise purity, rounding mode or noise-only validity. Other processing can also yield bounded residuals, and clipping would invalidate the assumed explanation. No gain/lag fit or threshold relaxation was performed.

All three source SHA256 values match before/after and agree with the frozen attempt03 record. Original attempt03 metrics/report and failed dev-additivity gate were not edited; this separate diagnosis supplies context only. No loss weight or later attempt was selected. Source paths, full quantiles, conditional-bound counts, versions, code/config hashes and timings are in metrics.json.

The single diagnostic is complete and stopped; Root decides any later action.
