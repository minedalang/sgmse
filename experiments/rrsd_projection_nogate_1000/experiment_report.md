# rrsd_projection_nogate_1000

Training completed for 1000 steps from the 80k EMA Router. Train loss epoch: 3075.516. Checkpoint: `joint_train_logs/rrhts7mb-rrsd_projection_nogate_1000/epoch=0-step=1000.ckpt`.

Inference used CUDA, EMA, RRSD PC, N=30, SNR=0.5, seed 2026 on the existing 32-utterance manifest. Enhanced wavs: `/tmp/rrsd_projection_nogate_eval`.

Means: CAM++ enhanced-clean cosine=0.841998, PESQ=1.693246, STOI=0.797569, SI-SDR=6.875934 dB, same-speaker CAM++=0.838324. DNSMOS P808=3.918293, SIG=3.425466, BAK=4.123460, OVRL=3.111670.

Paired bootstrap (20,000 resamples) against 80k Router and, for gate, no-gate control is recorded in `metrics.json`.

The no-gate last.ckpt save failed after training with Errno 28 because the data filesystem was full; the epoch checkpoint is complete and independently load-validated. The partial last checkpoint was moved to `/tmp/rrsd_failed_checkpoints/rrsd_projection_nogate_1000-last.ckpt.partial`.
