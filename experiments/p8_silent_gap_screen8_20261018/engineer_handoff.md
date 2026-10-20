# Worker handoff

Execute once from the repository:

```bash
/data/coding/sgmse/env/bin/python experiments/p8_silent_gap_screen8_20261018/screen.py
```

Prepared only: no acoustic screen, model inference, training, or audio download was executed. `source_snapshot.json` fixes the eight manifest index/basename/fileid mappings and hashes all 48 clean, noisy, Y_pre, Off, Router, and archived noise sources. The worker checks each hash before reading and after measurement. Source WAVs and existing experiment records remain unchanged.

Prior configuration inspected: `p8_local_content_probe_20261020/probe_local_content.py`, its config and completed summary, plus the subsequent waveform/envelope probe. Those analyzed clean-active half-second windows within 9–21 s, with overlapping 32 ms Hann frames and adaptive clean activity. This new fixed quiet-run configuration was not previously executed.

This screen uses float64 disjoint 20 ms rectangular frames over each 30 s source. Exact signal power contains no epsilon. First/last 0.2 s are excluded before run extraction, so only interior samples can qualify. Quiet runs require at least 15 consecutive frames, each at/below three PCM16 LSB RMS. Gap coordinates are half-open and sample exact. All qualifying gaps appear in the output, including unflagged ones. Full frame quiet/active/unclassified/boundary masks remain reviewable.

Ranking requires backend RMS strictly above -60 dBFS and gain at least 10 dB above Y_pre in that same gap. Zero baseline or numerator gives undefined gain and cannot pass the flag. This is a conservative exploratory ranking rule, with no speech or hallucination interpretation. Active complementary relative L2 uses clean frames at/above -70 dBFS; between-threshold frames remain unclassified. Delta waveform energy includes phase changes, so signed output-minus-Y_pre energy change is also reported.

10720 is the discovery item. `exploratory_other7` denotes items held out with respect to the tail hypothesis but previously included in global audits. They are not confirmation or virgin cross-validation. These labels are preserved in each gap, each utterance, and group aggregates.

Synthetic engineering checks passed syntax, exact-zero energy, run extraction at 15 versus 14 frames, zero denominator handling, and zero-error relative L2 handling. Source hashing is preparation, not an experiment result. The worker produces required `metrics.json` and `experiment_report.md` beside `config.yaml`, plus all-gap CSV, frame masks and provenance. Total generated and existing files are checked against a five million byte budget before runtime artifacts are written. Existing results are guarded with exclusive creation; `run_started.json` prevents concurrent or accidental repeat execution. On failure inspect `failure.json` and record a diagnosis before considering another attempt.
