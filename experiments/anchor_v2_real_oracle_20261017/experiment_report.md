# Anchor V2 clean-reference real-audio Oracle (32 files)

## Question

Does the two-direction residual subspace used by Anchor V2 contain final-trajectory quality headroom on real speech, before training a controller?

## Setup

- Checkpoint: `joint_train_logs/sb61i0gj_RRSD/epoch=118-step=80000.ckpt`, EMA enabled.
- Manifest: `experiments/rrsd_cv_repeat32/manifest.csv` (32 files, repeated-speaker CV diagnostic).
- CUDA PC sampler, `N=30`, `snr=0.5`, seed `2026`; each file uses a deterministic filename-derived seed shared by all arms.
- Arms: unchanged Router, `t_cut=0.05`, and clean-reference Anchor V2 Oracle.
- Oracle coefficients solve the bounded two-direction tile least-squares problem at every predictor step, then feed corrected `x0` through the complete reverse trajectory.
- Clean is used only to construct this diagnostic Oracle; it is not passed to the normal deployment sampler.
- Temporary wavs: `/dev/shm/anchor_v2_real_oracle_wavs_32`; no dataset, checkpoint, or production model path was changed.
- Metrics: PESQ-WB, STOI, SI-SDR, DNSMOS P808/SIG/BAK/OVRL. Paired bootstrap uses 20,000 resamples with seed 2026.

## 32-file means

| arm | PESQ | STOI | SI-SDR (dB) | DNSMOS P808 | DNSMOS SIG | DNSMOS BAK | DNSMOS OVRL |
|---|---:|---:|---:|---:|---:|---:|---:|
| Router | 1.702189 | 0.795823 | 6.954548 | 3.922351 | 3.408213 | 4.117465 | 3.089099 |
| t_cut=0.05 | 1.702004 | 0.795606 | 6.902938 | 3.928247 | 3.412476 | 4.117502 | 3.094026 |
| Anchor Oracle | 1.690745 | 0.795143 | 7.402782 | 3.836338 | 3.339785 | 4.105690 | 3.018148 |

## Paired bootstrap differences versus Router (95% CI)

| metric | t_cut=0.05 delta [95% CI] | Anchor Oracle delta [95% CI] |
|---|---:|---:|
| PESQ | -0.000185 [-0.002828, 0.002805] | -0.011443 [-0.039215, 0.019371] |
| STOI | -0.000217 [-0.000303, -0.000139] | -0.000680 [-0.002545, 0.001523] |
| SI-SDR (dB) | -0.051610 [-0.058397, -0.044444] | +0.448234 [+0.359928, +0.540420] |
| DNSMOS P808 | +0.005896 [+0.003141, +0.008741] | -0.086014 [-0.113909, -0.059525] |
| DNSMOS SIG | +0.004263 [+0.000693, +0.008967] | -0.068428 [-0.092893, -0.045387] |
| DNSMOS BAK | +0.000037 [-0.002381, +0.002626] | -0.011775 [-0.021844, -0.001605] |
| DNSMOS OVRL | +0.004926 [+0.000937, +0.010548] | -0.070951 [-0.095117, -0.047702] |

## Decision

The 32-file clean-reference Oracle confirms substantial SI-SDR headroom (`+0.448 dB`, CI strictly positive), but it lowers PESQ and all DNSMOS quality dimensions. OVRL decreases by `-0.070951` with a 95% CI of `[-0.095117, -0.047702]`, so the required perceptual headroom is absent. Do not train an Anchor controller from this result. Archive Anchor V2 and move to the next independent headroom test.

The complete per-file metrics and reproducibility metadata are stored in `metrics_32file.json`. The diagnostic script remains in this experiment directory; no wav files are copied into the repository.
