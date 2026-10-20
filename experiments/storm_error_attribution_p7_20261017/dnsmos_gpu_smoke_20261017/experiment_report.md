# DNSMOS GPU smoke — 2026-10-17

## Purpose and scope

This smoke checked whether DNSMOS can use the CUDA ONNX Runtime provider by default, whether the scores remain numerically stable against CPU, and how long the DNSMOS component takes on one existing 30-second utterance. It did not rerun the existing 32-utterance attribution evaluation.

## Implementation

Both `/data/coding/mymetrics/get_metrics_for_papertable_forDNS.py` and `experiments/storm_error_attribution_p7_20261017/frontend_attribution/evaluate_dnsmos.py` now default to `auto`: use CUDA when PyTorch CUDA is available and the two DNSMOS ONNX sessions include `CUDAExecutionProvider`; otherwise warn and fall back to CPU. Both retain an explicit CPU option. The P7 evaluator also records the resolved device and ONNX session providers in its report.

The DNSMOS mel features are still computed by librosa on CPU. PESQ, STOI, SI-SDR, file loading, and other evaluation work are also outside this GPU timing.

## Verification

- Environment: `/data/coding/mymetrics/env`, Torch 2.14.1+cu130, TorchMetrics 1.9.0, ONNX Runtime 1.31.0, NVIDIA L40S.
- Input: existing `frontend_only/000_fileid_2673.wav`, 16 kHz, 30 seconds; no source audio was written or modified.
- After one warm-up call per device, three repeated calls averaged 3.213 seconds on CPU and 0.199 seconds on CUDA, about 16.2× for this DNSMOS component on this utterance.
- ONNX profiling confirmed 41 node events from `sig_bak_ovr.onnx` and 21 from `model_v8.onnx` ran on `CUDAExecutionProvider`; no profiled model nodes used CPU fallback.
- CPU and CUDA scores were within 0.000131 maximum absolute difference. Each device returned identical scores across the three repeated calls.
- An ONNX Runtime warning about plugin-device registration appeared during CUDA session creation, but the session completed inference and the execution profile confirmed CUDA node execution.

## Limits and decision

This is a one-utterance smoke on a previously screened development split. The measured speedup applies only to DNSMOS, not the full metrics pipeline or an entire corpus. The earlier 32-utterance DNSMOS values remain CPU-computed and unchanged. No full evaluation, training, dataset modification, or checkpoint modification was performed.

The migration is ready for future evaluations using the mymetrics environment. Next P7 work is the separately planned fixed amplitude/phase cross-spectral diagnostic; this GPU migration check is complete.
