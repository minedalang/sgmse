# TR-0 short adapter screen — 2026-10-18

## Question

Does a small backend-only adapter behave differently when its one-step `x0` reconstruction error is constrained relative to the frozen 80k baseline?

## Protocol

The StoRM frontend, Router, and all original NCSN++ parameters were frozen. Only the 16,512-parameter zero-initialized residual adapter after `all_modules.33` was optimized. The `score_only` arm used the original score-matching loss. The `trust_region` arm added a clean-training-only hinge:

`relu(MSE(x0_adapter, clean) - MSE(x0_frozen80k, clean))`.

The baseline output was computed on the same sampled `x_t`, timestep, and conditioning, with the adapter temporarily disabled. Training used only the local `tr_synthetic` SCP, batch size 1, 200 steps, two seeds per arm, CUDA, and AdamW at `1e-4`. Only adapter state dictionaries were saved.

## Interpretation boundary

This is a short optimization screen, not an enhancement evaluation. It can show whether the two losses produce distinct, stable adapter updates, but it cannot establish DNSMOS, PESQ, STOI, SI-SDR, speaker, or content improvements. Fresh-CV inference is required next; the fresh manifest is `experiments/tr0_fresh_cv_manifest_20261018/manifest.csv`.

The original 80k checkpoint and dataset were not modified. No full checkpoint or W&B artifact was created.
