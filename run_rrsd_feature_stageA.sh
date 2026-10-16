#!/usr/bin/env bash
set -euo pipefail
cd /data/coding/sgmse
ROOT="${ROOT:-/data/coding/sgmse/experiment_logs/rrsd_tcut_screen}"
ENCODER="${ENCODER:-unispeech_sv}"
MODEL_PATH="${MODEL_PATH:-microsoft/unispeech-sat-base-plus-sv}"
DEVICE="${DEVICE:-cuda:0}"
COUNT="${COUNT:-32}"
MAX_PLM_LENGTH="${MAX_PLM_LENGTH:-32000}"
OUT="${OUT:-${ROOT}/feature_stageA/${ENCODER}}"
mkdir -p "$OUT"
for filename in "$ROOT/manifest.csv" "$ROOT/paired_metrics.csv"; do
  if [ ! -f "$filename" ]; then echo "[ERROR] missing $filename" >&2; exit 1; fi
done
for d in "$ROOT/off" "$ROOT/router" "$ROOT/clean"; do
  if [ ! -d "$d" ]; then echo "[ERROR] missing directory $d" >&2; exit 1; fi
done
printf '[RUN] Stage A: encoder=%s model=%s n=%s device=%s\n' "$ENCODER" "$MODEL_PATH" "$COUNT" "$DEVICE"
python rrsd_stageA_feature_probe.py \
  --root "$ROOT" --model-type "$ENCODER" --model-path "$MODEL_PATH" \
  --device "$DEVICE" --count "$COUNT" --max-plm-length "$MAX_PLM_LENGTH" \
  --out-dir "$OUT" 2>&1 | tee "$OUT/console.log"
