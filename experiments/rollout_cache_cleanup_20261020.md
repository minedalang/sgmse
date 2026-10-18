# Rollout cache cleanup (2026-10-20)

The two files below were intermediate 945 MiB CPU rollout-state caches used by completed rollout-aware adapter proxy pilots:

- `experiments/rollout_adapter_pilot_20261018/rollout_state_cache.pt`
- `experiments/rollout_adapter_replication_20261018/rollout_state_cache.pt`

They were deleted to recover approximately 1.9 GiB under the 2.2 GiB `/data` headroom constraint. The source audio, dataset, checkpoints, adapter states, per-file metrics, proxy metrics, configs, scripts, reports, and input/output hashes remain. The caches are reproducible from the saved scripts but are not required to interpret the archived results. This cleanup does not alter any dataset or checkpoint.

SHA-256 values before deletion are recorded in `rollout_cache_cleanup_20261020.sha256`.
