# Worker handoff

Prepared only; no inference executed. Separate bounded within-case/SNR replication on noise_minus20dB. Completed primary pilot files remain unchanged.

Run once from repository root:

```bash
PATH=/data/coding/sgmse/env/bin:$PATH /data/coding/sgmse/env/bin/python experiments/p8_10720_tail_condition20_20261018/run_pilot.py > experiments/p8_10720_tail_condition20_20261018/run.log 2>&1
```

Exactly two backend-off runs: no_op and condition_tail_clean, seed2997368683, NFE30. One model load, one frontend recomputation from original FLOAT20dB input. Exact20dB frontend float32 hash and exact saved20dB no_op PCM16 file hash are mandatory gates. Original float frontend is held globally unchanged in both runs; only noisy condition samples240800:252000 are interpolated toward clean using the original internal20ms cosine mask. Outside input samples and noisy global reference peak must remain identical.

Saves raw arrays and PCM16 outputs, frontend/mask/edited condition, code/checkpoint/input/RNG provenance, tail energy and outside/active output L2 summaries, raw fixed20ms frames.csv, metrics.json and experiment_report.md. Clean activity is fixed−70dBFS RMS on full16kHz reference, disjoint20ms frames. Raw output comparisons use new no_op raw; PCM metrics are reference only. Output cap20MB. Failure snapshots preserved; nonzero exit; no automatic retry.

Synthetic checks, Python parse, source code hashes, two-run design and unchanged original pilot driver hash verified in engineer_validation.json. This clean-reference oracle probes mechanism only; it is neither a trained method nor independent generalization evidence.
