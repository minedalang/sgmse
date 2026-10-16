#!/usr/bin/env bash
set -euo pipefail

ROOT="${SGMSE_ROOT:-/data/coding/sgmse}"
DATA_ROOT="${DATA_ROOT:-/data/datasets/dns_ins20/dns_snr_-20_-5/synthetic_snr-20_-5}"
SCP_ROOT="${SCP_ROOT:-$DATA_ROOT/scp_new}"
CKPT="${ROUTER_CKPT:-$ROOT/joint_train_logs/sb61i0gj_RRSD/epoch=118-step=80000.ckpt}"
STEPS="${STEPS:-1000}"
BATCH_SIZE="${BATCH_SIZE:-8}"
SEED="${SEED:-2026}"
SSL_WEIGHT="${SSL_WEIGHT:-10.0}"
TAG="${TAG:-rrsd_ssl_mse_w${SSL_WEIGHT}_s${STEPS}}"
LR="${LR:-3e-5}"
OBJECTIVE="${OBJECTIVE:-ssl_mse}"
MARGIN="${MARGIN:-0.0}"
RRSD_TEACHER="${RRSD_TEACHER:-future_grad}"
LOOKAHEAD_K="${LOOKAHEAD_K:-1}"
LOOKAHEAD_CANDIDATES="${LOOKAHEAD_CANDIDATES:-0,0.25,0.5,0.75,1.0}"
LOOKAHEAD_DT="${LOOKAHEAD_DT:-0.0}"
LOOKAHEAD_CHUNK="${LOOKAHEAD_CHUNK:-0}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export SGMSE_STORM_CHECKPOINT=1 SGMSE_STORM_CHECKPOINT_MODE=full
export WANDB_MODE=offline

cd "$ROOT"
mkdir -p experiment_logs
LOG="experiment_logs/${TAG}.log"
echo "[RRSD-SSL-MSE-TRAIN] ckpt=$CKPT steps=$STEPS batch=$BATCH_SIZE weight=$SSL_WEIGHT seed=$SEED teacher=$RRSD_TEACHER k=$LOOKAHEAD_K candidates=$LOOKAHEAD_CANDIDATES" | tee "$LOG"

PATH="$ROOT/env/bin:$PATH" "$ROOT/env/bin/python" -u joint_train.py \
  --backbone ncsnpp_v2 --sde ouve --loss_type score_matching \
  --num_eval_files 5 --loss_weighting 'sigma^2' \
  --c_in 1 --c_out 1 --c_skip 0 --network_scaling 1/t \
  --train_type mode2 --mode2_include_noisy_main \
  --nf 128 --ch_mult 1 2 2 2 --num_res_blocks 1 --attn_resolutions 0 \
  --selector_type none --dataset joint_scp \
  --train_wav_scp "$SCP_ROOT/tr_synthetic/wav.scp" \
  --valid_wav_scp "$SCP_ROOT/cv_synthetic/wav.scp" \
  --test_wav_scp "$SCP_ROOT/tt_real_recordings/wav.scp" \
  --clean_dir "$DATA_ROOT/clean" \
  --batch_size "$BATCH_SIZE" --accelerator gpu --devices 1 \
  --precision 32-true --max_steps "$STEPS" \
  --seed "$SEED" --lr "$LR" --wandb_name "$TAG" --save_ckpt_interval "$STEPS" \
  --frontend_trainable --frontend storm \
  --frontend_aux_loss spec_mse --frontend_nr_strategy auxiliary --frontend_nr_weight 0 \
  --rrsd_weight 1.0 --rrsd_hidden_ch 48 --rrsd_num_bands 32 --rrsd_num_time_blocks 16 \
  --rrsd_init_alpha 0.05 --rrsd_smooth_weight 0.01 --rrsd_oracle_eps 1e-6 \
  --rrsd_teacher "$RRSD_TEACHER" --rrsd_freeze_base \
  --rrsd_lookahead_k "$LOOKAHEAD_K" \
  --rrsd_lookahead_candidates "$LOOKAHEAD_CANDIDATES" \
  --rrsd_lookahead_dt "$LOOKAHEAD_DT" --rrsd_lookahead_chunk "$LOOKAHEAD_CHUNK" \
  --rrsd_desat_weight 0 \
  --contrastive_weight "$SSL_WEIGHT" \
  --contrastive_objective "$OBJECTIVE" --contrastive_margin "$MARGIN" \
  --contrastive_speaker_weight 0 \
  --plm_type unispeech_sat --plm_name_or_path microsoft/unispeech-sat-base-plus-sv \
  --contrastive_temperature 0.1 --contrastive_clean_weight 1.0 \
  --max_plm_length 32000 \
  --weighting_denoiser_to_score 0.5 \
  --storm_nf 128 --storm_ch_mult 1 2 2 2 --storm_num_res_blocks 1 --storm_attn_resolutions 0 \
  --init_ckpt "$CKPT" --init_ckpt_use_ema 2>&1 | tee -a "$LOG"

echo "Log saved: $LOG"
