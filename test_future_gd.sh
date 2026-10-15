cd /data/coding/sgmse

# 1. 指定已经训练好的 80k Router checkpoint
# 注意：这里必须改成服务器上真实存在的路径
ROUTER_CKPT="/data/coding/sgmse/joint_train_logs/sb61i0gj_RRSD/epoch=118-step=80000.ckpt"

# 2. 启用 Future-grad 诊断
export SGMSE_RRSD_DIAG_BATCHES=16
export SGMSE_RRSD_DIAG_EXIT=1
# 新增：有限差分实验
export SGMSE_RRSD_FD_PROBE=1
export SGMSE_RRSD_FD_EPS=0.01
# 新增：Smooth Oracle 对照
export SGMSE_RRSD_SMOOTH_PROBE=1
# 3. 显存设置
export SGMSE_STORM_CHECKPOINT=1
export SGMSE_STORM_CHECKPOINT_MODE=full
export SGMSE_SCORE_CHECKPOINT=0

# 4. 确认诊断补丁已应用
grep -n "SGMSE_RRSD_DIAG_BATCHES" sgmse/joint_model.py

# 5. 确认 checkpoint 文件存在
if [ ! -f "$ROUTER_CKPT" ]; then
    echo "错误：找不到 Router checkpoint: $ROUTER_CKPT"
else
    CUDA_VISIBLE_DEVICES=0 python joint_train.py \
      --backbone ncsnpp_v2 \
      --sde ouve \
      --loss_type score_matching \
      --loss_weighting 'sigma^2' \
      --c_in 1 \
      --c_out 1 \
      --c_skip 0 \
      --network_scaling 1/t \
      --train_type mode2 \
      --mode2_include_noisy_main \
      --nf 128 \
      --ch_mult 1 2 2 2 \
      --num_res_blocks 1 \
      --attn_resolutions 0 \
      --selector_type none \
      --dataset joint_scp \
      --train_wav_scp "/data/datasets/dns_ins20/dns_snr_-20_-5/synthetic_snr-20_-5/scp_new/tr_synthetic/wav.scp" \
      --valid_wav_scp "/data/datasets/dns_ins20/dns_snr_-20_-5/synthetic_snr-20_-5/scp_new/cv_synthetic/wav.scp" \
      --test_wav_scp "/data/datasets/dns_ins20/dns_snr_-20_-5/synthetic_snr-20_-5/scp_new/tt_real_recordings/wav.scp" \
      --clean_dir "/data/datasets/dns_ins20/dns_snr_-20_-5/synthetic_snr-20_-5/clean" \
      --batch_size 2 \
      --accelerator gpu \
      --devices 1 \
      --precision 32-true \
      --max_epochs 1 \
      --frontend_trainable \
      --frontend storm \
      --frontend_aux_loss spec_mse \
      --frontend_nr_strategy auxiliary \
      --frontend_nr_weight 0 \
      --weighting_denoiser_to_score 0.5 \
      --storm_nf 128 \
      --storm_ch_mult 1 2 2 2 \
      --storm_num_res_blocks 1 \
      --storm_attn_resolutions 0 \
      --rrsd_weight 1.0 \
      --rrsd_hidden_ch 48 \
      --rrsd_num_bands 32 \
      --rrsd_num_time_blocks 16 \
      --rrsd_init_alpha 0.05 \
      --rrsd_smooth_weight 0.01 \
      --rrsd_oracle_eps 1e-6 \
      --rrsd_teacher future_grad \
      --rrsd_freeze_base \
      --rrsd_validate \
      --init_ckpt "$ROUTER_CKPT" \
      --init_ckpt_use_ema \
      --lr 0 \
      --nolog
fi