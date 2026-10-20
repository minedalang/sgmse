#!/usr/bin/env bash
set -euo pipefail
cd /data/coding/sgmse
export PYTHONPATH=/data/coding/sgmse
export PATH=/data/coding/sgmse/env/bin:$PATH
/data/coding/sgmse/env/bin/python experiments/speaker_headroom_replication_64_20261017/lock_manifest.py
/data/coding/sgmse/env/bin/python experiments/speaker_headroom_replication_64_20261017/stage_inputs.py
/data/coding/sgmse/env/bin/python -u joint_enhance_by_dir.py --ckpt /data/coding/sgmse/joint_train_logs/sb61i0gj_RRSD/epoch=118-step=80000.ckpt --in_dir /data/coding/sgmse/experiment_logs/speaker_headroom_replication_64_20261017/inputs/off --out_dir /data/coding/sgmse/experiment_logs/speaker_headroom_replication_64_20261017/off --device cuda --train_type mode2 --sampler_type pc --corrector none --N 30 --snr 0.5 --use_ema --rrsd --rrsd_scale 0 --seed 2026
/data/coding/sgmse/env/bin/python -u joint_enhance_by_dir.py --ckpt /data/coding/sgmse/joint_train_logs/sb61i0gj_RRSD/epoch=118-step=80000.ckpt --in_dir /data/coding/sgmse/experiment_logs/speaker_headroom_replication_64_20261017/inputs/router --out_dir /data/coding/sgmse/experiment_logs/speaker_headroom_replication_64_20261017/router --device cuda --train_type mode2 --sampler_type pc --corrector none --N 30 --snr 0.5 --use_ema --rrsd --rrsd_scale 1 --seed 2026
/data/coding/sgmse/env/bin/python experiments/speaker_headroom_replication_64_20261017/validate_outputs.py
/data/coding/sgmse/env/bin/python experiments/speaker_headroom_replication_64_20261017/analyze_campp.py
/data/coding/mymetrics/env/bin/python experiments/speaker_headroom_replication_64_20261017/evaluate_quality.py
