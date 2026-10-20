# Matched backend-only 500-step control — 2026-10-18

## Status

The no-loss control arm completed 500 optimization steps. A read-only 64-speaker evaluation was subsequently completed. The paired speaker-loss arm was not started because the current research plan pauses this branch pending stronger evidence that the backend needs speaker supervision. This control is not a positive Speaker-loss result; it is a matched reference and stability check.

## Fixed setup

- Initialization: `joint_train_logs/sb61i0gj_RRSD/epoch=118-step=80000.ckpt`, loaded with its saved EMA state.
- Backend: NCSN++ v2, OUVE, `mode2`, score matching with `sigma^2` weighting, `network_scaling=1/t`.
- Frontend: StoRM frozen and in evaluation mode.
- Router: frozen; RRSD auxiliary loss disabled.
- Batch size 8, learning rate `3e-5`, seed 2026, 32-bit precision, 500 steps.
- Dataset manifests point to `/data/datasets/dns_ins20`; the dataset was not modified.
- W&B was offline; no cloud synchronization was performed.

## Outcome

- `global_step=500`, `epoch=0`.
- Logged epoch training loss: `5963.087`.
- Logged final step loss: `10814.631`.
- Training reached `max_steps=500` and exited normally.
- Checkpoint: `logs/None-speaker_backend_only_matched500_control/epoch=0-step=500-last.ckpt`.
- SHA256: `f42b60d0d25d6cfc0bfef3255ec3c0ef3926968367145c9f271eee1411106740`.

## Interpretation boundary

The loss values are not comparable to PESQ, DNSMOS, SI-SDR, or speaker verification, and no claim about enhancement quality or identity preservation can be made from this control alone. A matched speaker-loss arm would be required for the pre-registered comparison, but it remains paused under the current plan. The control checkpoint is preserved for possible future use and does not overwrite the original 80k checkpoint.

## Reproduction

The exact command is in `commands.sh`; the complete console output is in `control_console.log`; the complete configuration is in `config.yaml`. Checkpoint hashes are in `checkpoint_sha256.txt`.

## Locked evaluation (read-only)

The matched 64-speaker evaluation is archived under `control_eval_64/`. CAM++ control−off same-speaker cosine is +0.00262 (CI −0.00832 to +0.01453), AUC is +0.00852 (CI −0.00462 to +0.02486), and EER is −0.01736 (CI −0.06253 to +0.03046); all include zero. The paired-clean cosine contrast is +0.00146 (CI −0.00562 to +0.00847). PESQ/STOI/SI-SDR and per-utterance records are in the control evaluation directory. DNSMOS has only a single-file smoke result; no full three-arm DNSMOS claim is made.
