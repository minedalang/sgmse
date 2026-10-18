# Reproducible paper figures and claim matrix (2026-10-20)

## Purpose

This post-hoc analysis turns the locked final evaluation into paper-ready plots and a claim matrix. It does not rerun inference, modify the DNS Ins20 data, tune a threshold, or recompute the bootstrap intervals.

## Inputs

- `experiments/final_eval_full_20261018/metrics/paired_bootstrap.pretty.json`
- `experiments/final_eval_full_20261018/metrics/{off,router,tcut005}/{no_reverb,with_reverb}.csv`
- `experiments/final_eval_reverb_interaction_20261018_metrics.json`

The final evaluation uses 150 matched file IDs per split, with the protocol documented in the source experiment report. The figures preserve the source bootstrap intervals and per-utterance values.

## Outputs

- `metric_delta_by_split.{png,pdf}`: Router and fixed t-cut paired deltas by split.
- `router_off_pooled_ci.{png,pdf}`: pooled Router-versus-off bootstrap intervals.
- `reverb_interaction_ci.{png,pdf}`: matched-file Router interaction intervals.
- `sisdr_dnsmos_tradeoff.{png,pdf}`: per-utterance SI-SDR versus DNSMOS-OVRL deltas.
- `claim_matrix.md`: wording supported by the current evidence and explicit boundaries.
- `metrics.json`: source hashes and reproducibility metadata.

## Main interpretation

The plots support a conditional reconstruction benefit with a perception–reconstruction trade-off and a significant interaction with reverberation. They do not support a global perceptual improvement, a causal reverb gate, or a second validated algorithmic innovation.

## Reproduction

```bash
/data/coding/sgmse/env/bin/python experiments/paper_figures_claim_matrix_20261020/make_figures.py
```

