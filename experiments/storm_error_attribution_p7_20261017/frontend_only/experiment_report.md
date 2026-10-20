# P7-C Frontend-only Y_pre export

Exported 32 files from the existing 32-row manifest with checkpoint `joint_train_logs/sb61i0gj_RRSD/epoch=118-step=80000.ckpt`. The checkpoint frontend is `storm`; all 267 frontend state entries were verified byte-value equal to the model-loaded weights (no missing, extra, or mismatched tensors). The checkpoint EMA loaded with 8 tracked router parameters; the frontend remains loaded from the raw checkpoint state.

Preprocessing matches `joint_enhance_by_dir.py`: mono conversion as needed, 16 kHz input, per-file peak normalization with 1e-12 clamp, `frontend(y / norm) * norm`, and the same center crop/zero-pad length match. Outputs were stored as unclipped FLOAT WAVs. No training, quality evaluation, data changes, or checkpoint changes were performed.

Checkpoint SHA-256: `ab50510ca7fd80798b70f3f0b5365b17448a5f98c75d38755b82212ae2a119c9`. Manifest SHA-256: `77b64db9da6a1ff372a22dcda4a6316743133f383103aaffb3dbce2c671c178e`. Setup: 4.529s; forwards+writes: 10.177s; total: 14.706s. Per-file hashes, lengths and timing are in `metrics.json`.
