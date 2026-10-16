#!/usr/bin/env bash
set -euo pipefail
cd /data/coding/sgmse
ROOT="${ROOT:-/data/coding/sgmse/experiment_logs/rrsd_tcut_screen}"
MODEL_TYPE="${MODEL_TYPE:-unispeech_sv}"
MODEL_PATH="${MODEL_PATH:-microsoft/unispeech-sat-base-plus-sv}"
COUNT="${COUNT:-32}"
DEVICE="${DEVICE:-cuda:0}"
NAME="${NAME:-${MODEL_TYPE}}"
LOCAL_ONLY="${LOCAL_ONLY:-1}"
OUT="$ROOT/feature_ccr_paper/$NAME"
mkdir -p "$OUT"
EXTRA=()
if [[ "$LOCAL_ONLY" == "1" ]]; then EXTRA+=(--local-only); fi
python rrsd_ccr_ssl_feature_compare.py \
  --root "$ROOT" --model-type "$MODEL_TYPE" --model-path "$MODEL_PATH" \
  --device "$DEVICE" --count "$COUNT" --layers 3 6 9 12 --out "$OUT" \
  "${EXTRA[@]}" 2>&1 | tee "$OUT/console.log"
