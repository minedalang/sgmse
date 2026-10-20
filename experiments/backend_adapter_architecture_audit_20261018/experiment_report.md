# Backend adapter architecture audit — 2026-10-18

## Question

Can a future low-cost backend-only adapter be inserted into the locked 80k NCSN++ v2 Score network without changing the StoRM frontend, Router, SDE, or checkpoint contract?

## Method

This was a read-only checkpoint/state-dict audit. It loaded the trusted 80k checkpoint on CPU, recorded its SHA256 and hyperparameters, enumerated the NCSN++ `dnn` tensors, and estimated a rank-4 flattened-convolution LoRA budget for late decoder blocks. No source data, checkpoint, model code, or training state was changed.

## Findings

- The locked Score backbone is `ncsnpp_v2`, `nf=128`, `ch_mult=[1,2,2,2]`, one residual block per level, `OUVE`, score matching with `sigma^2` weighting, and `network_scaling=1/t`.
- The `dnn` state has 27,776,038 parameters across 271 tensors.
- The late decoder candidate is represented by `dnn.all_modules.31`, `.32`, and `.33`; the audit estimates the rank-4 flattened-convolution factors at roughly 0.19M parameters. This is a budget estimate, not an implemented adapter.
- The checkpoint contains no adapter keys. Any future adapter must be zero-initialized or otherwise preserve an exact no-op at initialization, and must keep the original Score state untouched.

## Interpretation boundary

This audit establishes implementation feasibility and a parameter budget only. It does not show that an adapter improves PESQ, DNSMOS, SI-SDR, STOI, content, or identity. A future smoke must compare an exact no-op adapter against the unchanged 80k model before any optimization, then use a matched backend-only no-loss control and an independently selected validation set.

## Decision

Do not train an adapter from this audit alone. The current plan still requires an independent, reproducible perceptual or distortion signal before P5a/P5b training. If that signal is obtained, the late-decoder zero-init candidate is the smallest code surface to test first.
