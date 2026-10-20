# RRSD n30 hard harm gate comparison

CUDA N=30 run on fileids 2673, 2972, 7247, 4431 using EMA checkpoint and seed 2026. Harm steps were 6/30, 6/30, 3/30, and 4/30 respectively (19/120 = 15.8%). Generated WAVs are under `experiment_logs/rrsd_harm_oracle_n30_4_gate`. Route/base MSE values are in `metrics.json`; paired PESQ/STOI/SI-SDR evaluator was not run because this script's output manifest is not wired to the existing evaluator.
