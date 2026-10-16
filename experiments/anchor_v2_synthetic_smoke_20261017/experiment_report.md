# Anchor V2 synthetic engineering smoke

## Research question

Can the existing Anchor V2 controller run as an inference-observable, band-time two-direction correction, remain an exact no-op at initialization, and receive a usable training gradient?

## Setup

- Synthetic complex tensors with shape `[B=2,C=1,F=32,T=24]`, seed 7.
- Controller: hidden channels 12, 8 frequency bands, 6 time blocks.
- Target: `base + 0.22 * u_pre - 0.13 * u_noisy +` small complex noise.
- No dataset, checkpoint, or source model parameters were modified.

## Results

- Base proxy MSE: `0.0654148608`.
- Two-direction tile least-squares Oracle proxy MSE: `0.0003833932`.
- Zero-initialized controller output was exactly equal to `x0_base`.
- Backpropagation reached the controller output layer; gradient L1 was `0.2632306218`.
- `single_shot_anchor_v2_smoke_test.py`, `pathsim_vA_smoke_test.py`, and Python compilation of the relevant modules all passed.

## Interpretation and limits

This validates tensor shapes, no-op initialization, and the synthetic gradient path only. It does not show that Anchor V2 improves PESQ, SI-SDR, STOI, or DNSMOS. A real-audio probe must use the existing 80k checkpoint, freeze the base model, compare against Router and `t_cut=0.05`, and use clean targets only for training-time oracle diagnostics.
