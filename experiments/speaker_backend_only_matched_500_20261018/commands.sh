#!/usr/bin/env bash
set -euo pipefail
ROOT=/data/coding/sgmse
DATA=/data/datasets/dns_ins20/dns_snr_-20_-5/synthetic_snr-20_-5
CKPT=$ROOT/joint_train_logs/sb61i0gj_RRSD/epoch=118-step=80000.ckpt
COMMON=(--backbone ncsnpp_v2 --sde ouve --loss_type score_matching --loss_weighting sigma^2 --c_in 1 --c_out 1 --c_skip 0 --network_scaling 1/t --train_type mode2 --mode2_include_noisy_main --nf 128 --ch_mult 1 2 2 2 --num_res_blocks 1 --attn_resolutions 0 --selector_type none --dataset joint_scp --train_wav_scp "$DATA/scp_new/tr_synthetic/wav.scp" --valid_wav_scp "$DATA/scp_new/cv_synthetic/wav.scp" --test_wav_scp "$DATA/scp_new/tt_real_recordings/wav.scp" --clean_dir "$DATA/clean" --batch_size 8 --accelerator gpu --devices 1 --precision 32-true --max_steps 500 --num_eval_files 0 --seed 2026 --lr 3e-5 --save_ckpt_interval 100000 --frontend storm --frontend_aux_loss spec_mse --frontend_nr_strategy auxiliary --frontend_nr_weight 0 --rrsd_weight 1.0 --rrsd_freeze_router --rrsd_disable_aux_loss --speaker_balanced_batches --plm_type unispeech_sat --plm_name_or_path microsoft/unispeech-sat-base-plus-sv --unispeech_sv_head --contrastive_temperature 0.1 --contrastive_clean_weight 1.0 --max_plm_length 32000 --weighting_denoiser_to_score 0.5 --storm_nf 128 --storm_ch_mult 1 2 2 2 --storm_num_res_blocks 1 --storm_attn_resolutions 0 --init_ckpt "$CKPT" --init_ckpt_use_ema)
export CUDA_VISIBLE_DEVICES=0 WANDB_MODE=offline SGMSE_WANDB_LOG_CODE=0
cd "$ROOT"
TAG=$1
shift
PATH="$ROOT/env/bin:$PATH" PYTHONPATH="$ROOT" "$ROOT/env/bin/python" -u joint_train.py "${COMMON[@]}" --wandb_name "$TAG" --log_dir "$ROOT/experiments/speaker_backend_only_matched_500_20261018/logs" "$@"
