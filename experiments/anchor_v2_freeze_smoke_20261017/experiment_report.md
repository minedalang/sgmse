# Anchor V2 controller-only freeze-path smoke

## Question

Does the new `--anchor_v2_freeze_base` path freeze the score network and StoRM frontend while keeping only the Anchor V2 controller trainable?

## Command

The run used the existing 80k EMA Router checkpoint, CUDA, batch size 1, `max_steps=4`, `lr=0`, `--anchor_v2_weight 1`, `--anchor_v2_freeze_base`, single-shot training around reverse progress 0.5, `--rrsd_weight 0`, and `--nolog`. The full command is preserved in `/tmp/anchor_v2_freeze_smoke4.log`.

## Results

- Exit code: `0`.
- Trainable parameters: only `anchor_v2_controller.*`, 21.7K parameters in 6 tensors.
- Frozen parameters: score network and frontend, approximately 55.1M parameters.
- EMA tracked tensors: 6, matching the controller only.
- Loss: step `350.959`; epoch `178.817`.
- No checkpoint was written because this was a `--nolog` smoke run.

## Limits

This verifies parameter selection and the training graph only. `lr=0` does not test a nonzero optimizer update or real-audio quality. No source data or existing checkpoint was modified.
