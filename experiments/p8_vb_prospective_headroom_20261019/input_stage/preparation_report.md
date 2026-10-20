# CPU input preparation: PREPARED ONLY

No actual source waveform decoding, mixing, model import, ASR, network, or resampling performed. Source byte hashes were checked. Synthetic tests use temporary generated signals only.

Scope: all 8 frozen registry utterances, 32 prescribed noisy conditions, 8 scaled-clean controls; historical two noises and [80000:80000+L) recipe unchanged. Full float64 mixture then floor PCM16 and float32 storage. Metadata-only expected signal artifact bytes: 13,730,360; below 24MiB stage and 64MiB parent budget. Stage uses 118s child deadline plus 2s final accounting. Parent geometry is exactly this experiment root, not experiments/. Stage and parent caps independently monitored with reserves.

Worker command after Root review:
```
python3 -B /data/coding/sgmse/experiments/p8_vb_prospective_headroom_20261019/input_stage/prepare_inputs.py
```
Then unique independent standard-library reconstruction:
```
python3 -B /data/coding/sgmse/experiments/p8_vb_prospective_headroom_20261019/input_stage/validate_inputs.py
```
Runtime emits config.yaml (already frozen), metrics.json, experiment_report.md, input_registry.json, and exclusive INPUT_PREPARATION_STARTED.json. Do not retry failures. Validator independently re-implements mixture/scale/quantization/seed and all 40 files; also reference ZIP bytes and historical decoded-noise hash, SNRs, float/PCM residuals and peaks. It does not import preparation functions. Successful old implementation was adapted; independent algorithm mirrors specified formula rather than claiming mathematical independence of the specification.

Source provenance: local16k registry audio, filename association with publisher labels only; spoken alignment/local conversion remain unproven. Prior failed 2LSB gate unchanged. No model-stage authorization. Metadata originals frozen by source_locks. Current preparation tests and hashes in sibling JSON records.
