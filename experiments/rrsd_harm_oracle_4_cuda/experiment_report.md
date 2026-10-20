# RRSD Harm Oracle four-file CUDA validation

Same command as the smoke run with `--max_files 4 --N 4 --device cuda --out_dir experiment_logs/rrsd_harm_oracle_4_cuda --use_ema`; checkpoint `joint_train_logs/sb61i0gj_RRSD/epoch=118-step=80000.ckpt`, seed 2026, commit `3e08d3c23bae552238dfcc4abb75c05a7e8b1966`.

All 4 files completed. Harm was detected on 4/16 predictor steps (25%, exactly 1/4 on each utterance). Mean per-file `(route MSE - base MSE)` was `-8.54e-7`; route and base feature errors were therefore very close, with no consistent increase across files. Per-utterance values are in `metrics.json` and generated WAVs are under `experiment_logs/rrsd_harm_oracle_4_cuda`.
