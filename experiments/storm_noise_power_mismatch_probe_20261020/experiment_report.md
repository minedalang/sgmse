# StoRM noise-power mismatch / predictor-kink probe

## Status: completed low-cost smoke (after one environment failure)

The first attempt did not execute. Importing `sgmse.joint_model` attempted to build the project's CUDA/C++ `upfirdn2d` extension and failed in `torch.utils.cpp_extension.verify_ninja_availability()` because `/data/coding/sgmse/env/bin` was absent from `PATH`. The Python package `ninja==1.13.2` was installed. That failed attempt is retained in `metrics.json`; it loaded no checkpoint and ran no inference.

The fixed rerun explicitly exported `PATH=/data/coding/sgmse/env/bin:$PATH`. Import/CUDA smoke passed (`NVIDIA L40S`, CUDA available), the EMA checkpoint loaded, and all 4 utterances × 5 M values completed in 216.94 s. No WAV files were written.

## Question and design

Read-only probe inspired by arXiv:2606.24035. On four utterances from four distinct speakers in the locked fresh-CV manifest (first occurrence of each speaker), construct `y_M = clean + M * (noisy - clean)` in memory for `M = [0.5, 0.75, 1.0, 1.25, 1.5]`. `M=1` reproduces the paired noisy input. This scales the noise residual while holding clean speech fixed; the per-file measured SNR is reported in `metrics.json`.

The frozen 80k EMA checkpoint and StoRM frontend are used. For every M, compare the frontend-only predictor output against Router-off StoRM PC output (`N=30`, `snr=0.5`, no corrector), with seed 2026. The mixture is peak-normalized before the frontend and restored afterward, matching the production inference path. The later StoRM call normalizes the frontend/noisy condition jointly. Such a global normalization does not erase M's change in clean-to-residual ratio; empirical input SNR is recorded to verify that.

## Scope and safeguards

- Four utterances / four speakers; no official final-set utterances.
- No new WAVs are written. Inputs and outputs stay in memory; only scalar metrics and hashes are saved.
- No training, checkpoint modification, or dataset modification.
- PESQ-WB, STOI, SI-SDR are reported per utterance and M for both frontend-only and Router-off output.
- SI-SDR first slopes and second finite differences over equally spaced M are reported as a descriptive kink/curvature statistic, not a proof of a predictor discontinuity.

## Confounds and limits

This small four-speaker pilot has low power. M changes effective noise level but preserves the particular noise waveform, so it does not sample independent noise realizations. PESQ/STOI/SI-SDR can vary non-monotonically with SNR; a sharp curve alone does not prove an internal predictor kink. This test does not measure DNSMOS or establish listening quality. It can only determine whether a reproducible, localized curvature signal is visible in these utterances and whether it originates in the frontend output or downstream diffusion refinement.

## Results

Mean measured input SNR decreased monotonically from -7.48 dB (`M=0.5`) to -17.02 dB (`M=1.5`). Router-off PC PESQ/STOI/SI-SDR also decreased monotonically: PESQ 2.051/1.809/1.669/1.592/1.508, STOI 0.886/0.846/0.818/0.796/0.776, and SI-SDR 10.145/7.817/6.550/5.593/4.869 for the five M values. The frozen frontend-only path showed the same monotonic trend.

The largest SI-SDR second differences occurred in the first interval (`M=0.5→0.75→1.0`) for 3/4 frontend cases and 3/4 Router-off cases; the remaining case also had its maximum at the first interval for Router-off. This is a smooth low-noise saturation/curvature pattern, not a localized discontinuity near `M=1`. Curvature magnitude was comparable in frontend-only and downstream paths (mean max absolute second difference 19.56 vs 16.96 dB/M²). The probe therefore shows noise-power sensitivity and smooth nonlinear response, but no reproducible predictor kink attributable to the diffusion stage. It does not justify mismatch-aware training or a new controller. See `metrics.json`, `detailed_metrics.json`, and `per_utterance_per_M.csv`; do not infer a general quality gain or start training from this probe alone.
