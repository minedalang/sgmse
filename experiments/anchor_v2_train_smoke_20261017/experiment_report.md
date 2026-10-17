# Anchor V2 CUDA smoke (2026-10-17)

The 4-step CUDA smoke completed successfully with exit code 0. It initialized from the 80k EMA Router checkpoint:

`/data/coding/sgmse/joint_train_logs/sb61i0gj_RRSD/epoch=118-step=80000.ckpt`

Configuration used `max_steps=4`, `batch_size=1`, `lr=0`, `seed=2026`, `rrsd_weight=0`, `anchor_v2_weight=1`, `anchor_v2_single_shot_train=true`, and `anchor_v2_single_shot_ratio=0.5`. Training and validation used the existing `cv_synthetic` SCP. The complete command was:

```bash
PATH=/data/coding/sgmse/env/bin:$PATH CUDA_VISIBLE_DEVICES=0 SGMSE_STORM_CHECKPOINT=1 SGMSE_STORM_CHECKPOINT_MODE=full /data/coding/sgmse/env/bin/python -u joint_train.py --nolog --backbone ncsnpp_v2 --sde ouve --loss_type score_matching --num_eval_files 0 --loss_weighting sigma^2 --c_in 1 --c_out 1 --c_skip 0 --network_scaling 1/t --train_type mode2 --mode2_include_noisy_main --nf 128 --ch_mult 1 2 2 2 --num_res_blocks 1 --attn_resolutions 0 --selector_type none --dataset joint_scp --train_wav_scp /data/datasets/dns_ins20/dns_snr_-20_-5/synthetic_snr-20_-5/scp_new/cv_synthetic/wav.scp --valid_wav_scp /data/datasets/dns_ins20/dns_snr_-20_-5/synthetic_snr-20_-5/scp_new/cv_synthetic/wav.scp --test_wav_scp /data/datasets/dns_ins20/dns_snr_-20_-5/synthetic_snr-20_-5/scp_new/cv_synthetic/wav.scp --clean_dir /data/datasets/dns_ins20/dns_snr_-20_-5/synthetic_snr-20_-5/clean --batch_size 1 --accelerator gpu --devices 1 --precision 32-true --max_steps 4 --seed 2026 --lr 0 --save_ckpt_interval 4 --log_dir /tmp/anchor_v2_smoke_logs --frontend_trainable --frontend storm --frontend_aux_loss spec_mse --frontend_nr_strategy auxiliary --frontend_nr_weight 0 --rrsd_weight 0 --rrsd_hidden_ch 48 --rrsd_num_bands 32 --rrsd_num_time_blocks 16 --rrsd_init_alpha 0.05 --rrsd_smooth_weight 0.01 --rrsd_oracle_eps 1e-6 --rrsd_teacher projection --anchor_v2_weight 1 --anchor_v2_single_shot_train --anchor_v2_single_shot_ratio 0.5 --weighting_denoiser_to_score 0.5 --storm_nf 128 --storm_ch_mult 1 2 2 2 --storm_num_res_blocks 1 --storm_attn_resolutions 0 --init_ckpt /data/coding/sgmse/joint_train_logs/sb61i0gj_RRSD/epoch=118-step=80000.ckpt --init_ckpt_use_ema
```

`AnchorV2Controller` was constructed with approximately 21.7K parameters. The final step loss was **350.959** and epoch loss was **178.817**. `--nolog` made Lightning use `enable_checkpointing=False`; no smoke checkpoint was written. Log: `/tmp/anchor_v2_smoke4.log`.
