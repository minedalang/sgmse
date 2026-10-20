# Conflict-weight dose comparison (0.001)

## Execution

The pre-registered 1000-step run completed successfully on GPU with batch size 1, seed 2026, learning rate 3e-5, and conflict weight 0.001. Checkpoint: `joint_train_logs/01maaifx-conflict_w0.001_s1000/epoch=0-step=1000.ckpt`. Training log: `experiment_logs/rrsd_conflict_train_conflict_w0.001_s1000.log`.

The first inference attempt failed before writing audio because the current router implementation receives `rrsd_t_max=None` and calls `float(None)`. This failure is recorded in the log. Inference was rerun with explicit `--rrsd_t_max 1.0`, which is the full time range used by the existing controls; all 32 files completed.

After this run, the `None` handling was fixed in `sgmse/joint_model.py` by passing the resolved SDE upper bound to the router. A one-file default-argument smoke test then completed successfully under `experiment_logs/rrsd_tmax_none_fix_smoke2/`.

## Metrics

| arm | PESQ | STOI | SI-SDR (dB) | DNSMOS P808 | SIG | BAK | OVRL |
|---|---:|---:|---:|---:|---:|---:|---:|
| conflict weight 0.001 | 1.751130 | 0.831884 | 7.744541 | 4.024486 | 3.479474 | 4.143871 | 3.187931 |
| existing control (weight 0) | 1.749303 | 0.831909 | 7.773258 | 4.023503 | 3.478363 | 4.143113 | 3.186280 |
| existing conflict (weight 0.0002) | 1.749728 | 0.831891 | 7.762543 | 4.024348 | 3.478721 | 4.143809 | 3.187303 |

Paired bootstrap 95% CIs for weight 0.001 minus control and minus existing 0.0002 are in `metrics.json`. Primary deltas versus control: PESQ +0.001827 [-0.006839, +0.011121], STOI -0.000025 [-0.000063, +0.000018], SI-SDR -0.028717 [-0.033579, -0.023898] dB. Versus 0.0002: PESQ +0.001403 [-0.002588, +0.007232], STOI -0.000006 [-0.000033, +0.000022], SI-SDR -0.018002 [-0.020392, -0.015632] dB. DNSMOS OVRL change is +0.001651 [0.000154, 0.003167] versus control and +0.000628 [-0.000585, +0.001910] versus 0.0002.

These dose differences are small on PESQ/STOI/DNSMOS; the 0.001 arm has a small but consistently negative paired SI-SDR shift relative to both existing arms.

## Artifacts

- New inference audio: `experiment_logs/rrsd_conflict_weight001_1000/infer/`
- Per-file evaluator output: `experiment_logs/rrsd_conflict_weight001_1000/metrics/metrics.csv`
- Training/inference logs: `experiment_logs/rrsd_conflict_train_conflict_w0.001_s1000.log`, `experiment_logs/rrsd_conflict_weight001_1000_infer.log`
