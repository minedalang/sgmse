# Worker handoff

Code prep only; no inference performed. Exactly two backend-off runs on original3321 PCM16 input: no_op, then condition_tail_clean. Seed513132500 is derived from base2026 plus little-endian4-byte BLAKE2b of the original exported06_ basename, modulo2**32. NFE30, one model load and one float frontend recomputation. Source manifest index6 identifies full30s/480000-sample utterance.

Run once from repository root:

```bash
PATH=/data/coding/sgmse/env/bin:$PATH /data/coding/sgmse/env/bin/python experiments/p8_3321_tail_condition_20261018/run_pilot.py > experiments/p8_3321_tail_condition_20261018/run.log 2>&1
```

Mandatory stop gates: source and current core code hashes match fixed records and successful10720 control; checkpoint SHA matches; freshly computed float frontend serialized to PCM16 matches existing3321 y_pre file byte hash; no_op matches existing3321 off PCM16 byte hash before intervention. No historical float frontend hash exists, so the new float hash is recorded without claiming such verification. Original float frontend is held globally unchanged. Condition-only intervention toward stored clean occupies samples361920:378560 (22.62–23.66s), with internal320-sample20ms raised cosine fades. Global noisy normalization peak and all samples outside that window must remain unchanged.

Outputs retain raw+PCM16, float frontend and its serialized verification WAV, mask/edited condition, fixed−70dBFS clean20ms frame activity, raw frame CSV, tail RMS/highband energy, outside and full active relative L2 comparisons against new no_op raw, provenance/config/metrics/report.25MB cap and exclusive run guard. Failures archive snapshot and stop; no automatic retry.

Synthetic tensor/mask/no-op/FFT checks, parsing, source hashes and relevant current code hashes passed. This preidentified picked second-utterance exploratory positive case is not unbiased population proof. Clean-reference intervention is an OOD mechanism probe, not a deployable or trained method. Existing pilot/results remain unchanged.
