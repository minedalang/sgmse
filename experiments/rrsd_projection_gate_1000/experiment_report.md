# rrsd_projection_gate_1000

Training completed for 1000 steps from the 80k EMA Router. Train loss epoch: 3131.678. Checkpoint: `joint_train_logs/d1z5ovza-rrsd_projection_gate_1000/epoch=0-step=1000.ckpt`.

Inference used CUDA, EMA, RRSD PC, N=30, SNR=0.5, seed 2026 on the existing 32-utterance manifest. Enhanced wavs: `/tmp/rrsd_projection_gate_eval`.

Means: CAM++ enhanced-clean cosine=0.844709, PESQ=1.677907, STOI=0.796927, SI-SDR=6.493572 dB, same-speaker CAM++=0.838629. DNSMOS P808=3.942807, SIG=3.445500, BAK=4.117770, OVRL=3.129861.

Paired bootstrap (20,000 resamples) against 80k Router and, for gate, no-gate control is recorded in `metrics.json`.

Explicit gate and strength losses were enabled as requested; the 4-step CUDA smoke passed before this run.

DNSMOS paired CI against Router is unavailable strictly: the available Router DNSMOS CSV has a different fileid manifest/order. Descriptive unpaired mean differences versus Router are P808 -0.079054, SIG -0.032206, BAK -0.025002, OVRL -0.055538; these are not paired estimates.
