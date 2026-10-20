# Corrected attempt02 handoff

Diagnosis: PyYAML1.1 converted original unquoted source_sha256 key off and backend value off into boolean False. Original failure is KeyError('off') at source hash lookup, before load_from_checkpoint. Failed metrics have empty runs and no loaded_model record. No model was loaded and no inference was reached in that failed attempt. Original captured driver/config/metrics/failure snapshot hashes remain preserved in engineer_validation.json.

Corrected config quotes 'off', validates all five source keys (clean,noisy,off,y_pre,noise) as strings, and strictly checks backend string and rrsd/use_ema boolean types. Archived paired noise is hashed and recorded for provenance only; it is not used to construct any input. CPU parser tests reject boolean key/value regressions. Model/frontend/backend execution stages are recorded; failed reports describe actual stages instead of claiming completed inference.

One corrected attempt authorized; execute exactly once from repository root:

```bash
PATH=/data/coding/sgmse/env/bin:$PATH /data/coding/sgmse/env/bin/python experiments/p8_3321_tail_condition_20261018/attempt02/run_pilot.py > experiments/p8_3321_tail_condition_20261018/attempt02/run.log 2>&1
```

Original global audit/source paths remain fixed and valid for this nested attempt. Two original3321 off runs only, seed513132500, NFE30, window361920:378560, fade320,25MB cap. Frontend PCM16 and no_op backend exact replay gates remain mandatory; original float frontend held unchanged, condition outside window and normalization peak unchanged. All original oracle/OOD and selected exploratory case limitations apply. No engineer inference performed. No automatic retry after any further failure.

Parser verification command:

```bash
PATH=/data/coding/sgmse/env/bin:$PATH /data/coding/sgmse/env/bin/python experiments/p8_3321_tail_condition_20261018/attempt02/test_config.py
```
