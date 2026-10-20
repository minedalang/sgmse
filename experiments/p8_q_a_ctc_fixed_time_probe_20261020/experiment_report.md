# P8-Q-A fixed-time CTC posterior probe (2026-10-20)

> **Indexing defect found during review on 2026-10-18:** the run completed,
> but seconds-to-frame `floor/ceil` conversion included one extra frame in
> 10/89 targets (50/445 condition rows), including two `/k/` targets.
> The original CSV and metrics are preserved for provenance. Its values are
> provisional and must not be used as the corrected fixed-time result.
> The integer-frame correction is tracked separately in
> `experiments/p8_q_a_ctc_fixed_time_v2_20261018/`.

## Purpose and protocol

The prior fixed-target Viterbi experiment allowed each condition to move a
token to a different time. This probe keeps each clean-derived token's clean
CTC frame interval fixed for all five conditions, and reports the posterior
there plus a nearby +/-200 ms maximum and mean. It reuses the existing local
wav2vec2 model and aligned 20 clips. It performs no training and does not
modify source audio, datasets, or checkpoints.

The token sequence and timestamps remain model-derived pseudo-labels. The
measurement therefore tests acoustic-model response at a fixed coordinate;
it is not a phoneme truth, transcript, WER/CER, or human recoverability test.

## Interpretation rule

The fixed-time value is the primary quantity. A high nearby maximum with a
low fixed-time value indicates that the model response may have shifted in
time; it must not be described as local restoration. Results are exploratory
and are not used as training targets or a deployment gate.

## Reproduction

```bash
/data/coding/sgmse/env/bin/python \
  experiments/p8_q_a_ctc_fixed_time_probe_20261020/fixed_time_probe.py
```

The command writes `metrics.json` and `fixed_time_token_posteriors.csv` in
this directory. The final report should be completed from those generated
values after the delegated smoke run.
