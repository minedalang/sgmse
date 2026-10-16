#!/usr/bin/env bash
# Read-only, no source patch; uses joint_train.py only for dataloader and state.
set -euo pipefail
ROOT="${SGMSE_ROOT:-/data/coding/sgmse}"
DATA_ROOT="${DATA_ROOT:-/data/datasets/dns_ins20/dns_snr_-20_-5/synthetic_snr-20_-5}"
SCP_ROOT="${SCP_ROOT:-$DATA_ROOT/scp_new}"
CKPT="${ROUTER_CKPT:-$ROOT/joint_train_logs/sb61i0gj_RRSD/epoch=118-step=80000.ckpt}"
BATCHES="${BATCHES:-4}"
BATCH_SIZE="${BATCH_SIZE:-1}"
PROBE_SPLIT="${PROBE_SPLIT:-cv_synthetic}"
PRECISION="${PRECISION:-32-true}"
SEED="${SEED:-2026}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export SGMSE_STORM_CHECKPOINT=1 SGMSE_STORM_CHECKPOINT_MODE=full
# Disable all previous expensive probes.
export SGMSE_RRSD_DIAG_BATCHES=0 SGMSE_RRSD_DIAG_EXIT=0
export SGMSE_RRSD_FD_PROBE=0 SGMSE_RRSD_SMOOTH_PROBE=0
export SGMSE_RRSD_SWEEP_BATCHES=0
export SGMSE_RRSD_KSTEP_BATCHES=0 SGMSE_RRSD_KSTEP_EXIT=0
export PG_BATCHES="$BATCHES"
export PG_MODEL_PATH="${PG_MODEL_PATH:-microsoft/wavlm-base-plus}"
export PG_LAYER="${PG_LAYER:-3}"
export PG_ANCHOR="${PG_ANCHOR:-zero}"
export PG_LOCAL_ONLY="${PG_LOCAL_ONLY:-1}"
export PG_T_MIN="${PG_T_MIN:-0.05}" PG_T_MAX="${PG_T_MAX:-0.5}"
export PG_FD_BATCHES="${PG_FD_BATCHES:-4}" PG_FD_EPS="${PG_FD_EPS:-0.02}"
for f in "$CKPT" "$SCP_ROOT/$PROBE_SPLIT/wav.scp" "$SCP_ROOT/cv_synthetic/wav.scp" "$SCP_ROOT/tt_real_recordings/wav.scp"; do
  [[ -f "$f" ]] || { echo "Missing file: $f" >&2; exit 2; }
done
[[ -d "$DATA_ROOT/clean" ]] || { echo "No clean directory: $DATA_ROOT/clean" >&2; exit 2; }
cd "$ROOT"
python -m py_compile rrsd_pg_agreement_probe.py joint_train.py sgmse/joint_model.py
mkdir -p experiment_logs
LOG="experiment_logs/rrsd_pg_${PG_ANCHOR}_wavlm_l${PG_LAYER}_${PROBE_SPLIT}_${BATCHES}b_seed${SEED}.log"
echo "[PG-RUN] ckpt=$CKPT wavlm=$PG_MODEL_PATH layer=$PG_LAYER batches=$BATCHES bs=$BATCH_SIZE t=[$PG_T_MIN,$PG_T_MAX) fp32=$PRECISION" | tee "$LOG"
python -u rrsd_pg_agreement_probe.py \
  --backbone ncsnpp_v2 --sde ouve --loss_type score_matching \
  --num_eval_files 5 --loss_weighting 'sigma^2' \
  --c_in 1 --c_out 1 --c_skip 0 --network_scaling 1/t \
  --train_type mode2 --mode2_include_noisy_main \
  --nf 128 --ch_mult 1 2 2 2 --num_res_blocks 1 --attn_resolutions 0 \
  --selector_type none --dataset joint_scp \
  --train_wav_scp "$SCP_ROOT/$PROBE_SPLIT/wav.scp" \
  --valid_wav_scp "$SCP_ROOT/cv_synthetic/wav.scp" \
  --test_wav_scp "$SCP_ROOT/tt_real_recordings/wav.scp" \
  --clean_dir "$DATA_ROOT/clean" \
  --batch_size "$BATCH_SIZE" --accelerator gpu --devices 1 \
  --precision "$PRECISION" --max_steps "$((BATCHES+2))" \
  --seed "$SEED" --lr 0 --nolog \
  --frontend_trainable --frontend storm \
  --frontend_aux_loss spec_mse --frontend_nr_strategy auxiliary --frontend_nr_weight 0 \
  --rrsd_weight 1.0 --rrsd_hidden_ch 48 \
  --rrsd_num_bands 32 --rrsd_num_time_blocks 16 \
  --rrsd_init_alpha 0.05 --rrsd_smooth_weight 0.01 --rrsd_oracle_eps 1e-6 \
  --rrsd_teacher future_grad --rrsd_freeze_base \
  --rrsd_desat_weight 0 \
  --weighting_denoiser_to_score 0.5 \
  --storm_nf 128 --storm_ch_mult 1 2 2 2 \
  --storm_num_res_blocks 1 --storm_attn_resolutions 0 \
  --init_ckpt "$CKPT" --init_ckpt_use_ema 2>&1 | tee -a "$LOG"
echo "Log saved: $LOG"
