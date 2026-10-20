# P7-C fixed amplitude/phase cross diagnostic — 2026-10-17

**Status:** complete

## Scope

This zero-training diagnostic reuses the existing 32-file P7-C development manifest. It constructs exactly two fixed arms from the paired Frontend-only waveform `Y_pre` and Router-off waveform `X_gen`: `|P|∠G` (`magP_phaseG`) and `|G|∠P` (`magG_phaseP`), where `P=STFT(Y_pre)` and `G=STFT(X_gen)`. No mixing coefficient, oracle selector, threshold, or model parameter was searched.

The split has already been used for development screening and is not an independent blind test. Clean audio was used only as the reference for offline metric evaluation.

## Construction and checks

- STFT: 16000 Hz, `n_fft=510`, `hop=128`, periodic Hann, centered; same Torch STFT/iSTFT definition as the prior P7-C analysis.
- Both arms were saved as 32-bit floating point WAVs after iSTFT. Sample lengths and finite values were checked per file.
- Original Frontend and Router-off iSTFT round-trip errors and each cross-arm STFT projection consistency residual are included in `metrics.json` and `per_utterance_metrics.csv`.
- DNSMOS request: `auto`; resolved runtime/provider: `{"device": "cuda:0", "execution_provider": "CUDAExecutionProvider", "onnx_session_providers": {"sig_bak_ovr": ["CUDAExecutionProvider", "CPUExecutionProvider"], "model_v8": ["CUDAExecutionProvider", "CPUExecutionProvider"]}}`.
- This run reuses the existing DNSMOS evaluator implementation and did not modify it.

## Mean scores

| Arm | PESQ-WB | STOI | SI-SDR (dB) | DNSMOS P.808 | SIG | BAK | OVRL |
|---|---:|---:|---:|---:|---:|---:|---:|
| magP_phaseG | 1.5159 | 0.8242 | 7.9016 | 3.7418 | 3.3502 | 4.0912 | 3.0423 |
| magG_phaseP | 1.7623 | 0.8327 | 7.9371 | 3.9834 | 3.4332 | 4.1143 | 3.1333 |
| frontend | 1.4937 | 0.8216 | 8.0894 | 3.6828 | 3.3062 | 4.0755 | 2.9958 |
| off | 1.7292 | 0.8310 | 7.2785 | 4.0447 | 3.4980 | 4.1439 | 3.2065 |
| router | 1.7490 | 0.8319 | 7.8010 | 4.0219 | 3.4777 | 4.1428 | 3.1854 |

The `metrics.json` contains paired per-utterance deltas and 10,000-replicate bootstrap 95% intervals for each new arm against each of Frontend-only, Router-off, and Router outputs.

## Waveform and STFT consistency

- `source_frontend_relative_rmse`: mean 1.243e-07, median 1.244e-07, max 1.262e-07.
- `source_off_relative_rmse`: mean 1.245e-07, median 1.243e-07, max 1.263e-07.
- `magP_phaseG_stft_projection_relative_l2`: mean 5.966e-02, median 6.293e-02, max 9.474e-02.
- `magG_phaseP_stft_projection_relative_l2`: mean 1.130e-01, median 1.051e-01, max 2.263e-01.

## Interpretation limits

These fixed crossings test whether the observed amplitude and phase strengths combine into useful waveforms under a specific iSTFT. They do not establish a causal mechanism, deployable rule, or generalization beyond this previously screened 32-utterance split. Any score change must be interpreted across all metrics; clean and DNSMOS were not used to select or tune an arm. No training, dataset edits, or checkpoint edits were performed.

## Paired comparison and decision

Against Router-off, `magP_phaseG` improved SI-SDR by 0.623 dB (paired bootstrap 95% CI [0.448, 0.804]) but reduced PESQ-WB by 0.213 ([-0.283, -0.145]), STOI by 0.00677 ([-0.01182, -0.00157]), and DNSMOS OVRL by 0.164 ([-0.223, -0.110]).

`magG_phaseP` improved PESQ-WB by 0.0331 ([0.0204, 0.0518]), STOI by 0.00171 ([0.00117, 0.00227]), and SI-SDR by 0.659 dB ([0.575, 0.739]) against Router-off, while DNSMOS OVRL fell by 0.0732 ([-0.1077, -0.0420]); P.808, SIG, and BAK also fell. Against Router, this arm retained small PESQ/STOI/SI-SDR gains but lowered OVRL by 0.0521 ([-0.0826, -0.0251]). No crossing dominates across metrics.

The target cross-spectra are not exactly STFT-consistent after iSTFT: mean relative projection residual is 5.97% for `magP_phaseG` and 11.30% for `magG_phaseP`. Thus, the evaluated WAVs contain the unavoidable consistency projection induced by iSTFT, rather than an exact realization of every desired time-frequency bin. This is a material limit when interpreting the metric changes.

**Decision:** the fixed swap confirms metric-specific trade-offs but provides no cross-metric evidence for a single robustly superior amplitude/phase assignment. Keep this as an exploratory diagnosis; do not infer a deployable selector or start training from this split.

**Runtime note:** ONNX Runtime emitted its CUDA plugin-device registration warning at session startup. The GPU-auto evaluator resolved to `cuda:0` for every row, both DNSMOS sessions listed CUDAExecutionProvider, and this environment's prior one-file profile observed CUDA graph execution. This run did not create a separate ONNX node profile for each utterance.
