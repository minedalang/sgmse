# RRSD value calibration pilot 1

The pilot completed successfully with one file and eight reverse diffusion steps using EMA checkpoint parameters. The runner wrote eight JSONL state records. Best candidate was 0.0 at steps 0, 1, and 7; it was 1.0 at steps 2 through 6. Relative gain increased with candidate value through steps 2–6, peaking at 0.1233 for candidate 1.0 at step 6; all candidates tied at steps 0 and 7. Wall time was 62.861 seconds and process exit code was 0. GPU after execution: NVIDIA L40S, 0 MiB / 49140 MiB, 20% utilization.

A first wrapper attempt failed because `/usr/bin/time` is unavailable, and a second failed because the runner requires a non-existent output path. Both failed directories were archived under `experiment_logs/rrsd_value_calibration_pilot1_failed_*`; the requested pilot then ran successfully without modifying code, data, or checkpoints.
