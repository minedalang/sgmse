# RRSD Harm Oracle CUDA smoke

Command used `PATH=/data/coding/sgmse/env/bin:$PATH CUDA_VISIBLE_DEVICES=0 env/bin/python -u rrsd_harm_oracle_infer.py ... --max_files 1 --N 4 --device cuda --out_dir experiment_logs/rrsd_harm_oracle_smoke_cuda_2 --use_ema`.

Checkpoint: `joint_train_logs/sb61i0gj_RRSD/epoch=118-step=80000.ckpt`; commit `3e08d3c23bae552238dfcc4abb75c05a7e8b1966`.

EMA loaded successfully. One utterance completed in four predictor steps, with 1 harm step (1/4). Route/base WavLM layer-3 feature MSE values are in `metrics.json`. A CPU attempt was aborted because inference was prohibitively slow; CUDA completed successfully.
