# Frozen final metrics reproducibility audit (2026-10-20)

## Purpose

This is a read-only recomputation of the means and paired Router-minus-off deltas used in the paper tables. It consumes the six per-utterance CSVs already archived by `final_eval_full_20261018`; it does not regenerate audio, recompute bootstrap intervals, tune a threshold, or read the raw dataset.

## Checks

- All six arm/split CSVs contain 150 rows.
- The three arms have identical `utt` ordering within each split.
- Every split/arm/metric mean agrees with `paired_bootstrap.pretty.json` to absolute error below `2e-10`.
- Every pooled Router-minus-off mean agrees with the archived JSON to absolute error below `2e-10`.
- Per-file CSV SHA256 values are recorded in `metrics.json`.

## Interpretation

The published table values are reproducible from the archived per-utterance results. This audit confirms arithmetic consistency; it does not add an independent test set or change the scientific interpretation.

## Reproduction

```bash
/data/coding/sgmse/env/bin/python experiments/final_metrics_repro_audit_20261020/audit_metrics.py
```
