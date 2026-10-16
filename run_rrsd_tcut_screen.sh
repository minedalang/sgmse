#!/usr/bin/env bash
set -euo pipefail
cd /data/coding/sgmse

# These are the same paths used in your previous validated RRSD diagnostics.
CKPT="${CKPT:-/data/coding/sgmse/joint_train_logs/sb61i0gj_RRSD/epoch=118-step=80000.ckpt}"
DNS="${DNS:-/data/datasets/dns_ins20/dns_snr_-20_-5/synthetic_snr-20_-5}"
SCP="${SCP:-${DNS}/scp_new/cv_synthetic/wav.scp}"
CLEAN="${CLEAN:-${DNS}/clean}"
OUT="${OUT:-/data/coding/sgmse/experiment_logs/rrsd_tcut_screen}"
COUNT="${COUNT:-32}"
SEED="${SEED:-2026}"
DEVICE="${DEVICE:-0}"

for path in "$CKPT" "$SCP"; do
  if [ ! -f "$path" ]; then echo "[ERROR] File not found: $path" >&2; exit 1; fi
done
if [ ! -d "$CLEAN" ]; then echo "[ERROR] Clean dir not found: $CLEAN" >&2; exit 1; fi
mkdir -p "$OUT"
python rrsd_tcut_prepare_subset.py --scp "$SCP" --clean-dir "$CLEAN" --out "$OUT" --count "$COUNT" --seed "$SEED"

for ARM in off router cut005 cut010; do
  case "$ARM" in
    off) SCALE=0; CUT=0 ;;
    router) SCALE=1; CUT=0 ;;
    cut005) SCALE=1; CUT=0.05 ;;
    cut010) SCALE=1; CUT=0.10 ;;
  esac
  mkdir -p "$OUT/$ARM"
  echo "[RUN] arm=$ARM scale=$SCALE t_cut=$CUT n=$COUNT"
  CUDA_VISIBLE_DEVICES="$DEVICE" python joint_enhance_by_dir.py \
    --ckpt "$CKPT" \
    --in_dir "$OUT/noisy" \
    --out_dir "$OUT/$ARM" \
    --frontend storm \
    --train_type mode2 \
    --sampler_type pc \
    --corrector none \
    --N 30 \
    --snr 0.5 \
    --use_ema \
    --rrsd \
    --rrsd_scale "$SCALE" \
    --rrsd_t_cut "$CUT" \
    --seed "$SEED" \
    2>&1 | tee "$OUT/${ARM}.log"
done

python rrsd_tcut_paired_eval.py --root "$OUT" | tee "$OUT/paired_summary.txt"
echo "[DONE] $OUT"
