# P8-Q-A CTC boundary audit (2026-10-20)

## Purpose

The fixed-target CTC pilot holds the clean-derived token sequence fixed, but
its Viterbi path is recomputed separately for every condition. This audit
measures the midpoint displacement of each condition's token path from the
clean path. A large displacement means that a target posterior is not a
same-time local phoneme measurement.

This is a read-only post-processing audit of
`p8_q_a_ctc_forced_alignment_20261020/token_alignments.csv`. It performs no
model inference, training, or data modification.

## Results

| condition | tokens | median absolute shift | p90 absolute shift | within +/-40 ms |
|---|---:|---:|---:|---:|
| clean | 89 | 0 ms | 0 ms | 100.0% |
| noisy | 89 | 120 ms | 652 ms | 38.2% |
| `Y_pre` | 89 | 20 ms | 240 ms | 66.3% |
| Router-off | 89 | 20 ms | 184 ms | 57.3% |
| Router | 89 | 20 ms | 240 ms | 57.3% |

For consonant tokens, the noisy median absolute shift is 140 ms; the
corresponding medians for `Y_pre`, Router-off, and Router are all 20 ms.
Several noisy `/d/` and vowel tokens move 600--820 ms, and one noisy `/n/`
token moves 820 ms. These are sequence-level realignments, not evidence that
the token was acoustically restored at its clean timestamp.

## Consequence for interpretation

The original forced-target posterior table remains a valid exploratory
sequence-level response of the CTC model, but it cannot be used as a
same-time local phoneme recovery measure without restricting analysis to
stable boundaries. In particular, the apparent increases for individual
`/d/` or `/k/` tokens may come from a different time location, a nearby
token, or a model-generated acoustic pattern. The boundary audit therefore
downgrades all event-level E1 interpretations from that pilot to unknown.

Future local phoneme analysis must either use independently verified event
timestamps or report only tokens whose condition-specific path stays within a
pre-registered tolerance. Clean-derived CTC labels remain unsuitable as
training supervision or as a claim of correct content recovery.

## Reproduction

```bash
/data/coding/sgmse/env/bin/python \
  experiments/p8_q_a_ctc_boundary_audit_20261020/boundary_audit.py
```

The script reads the prior CSV and writes `boundary_shifts.csv` and
`metrics.json` in this directory.
