# RRSD speaker SV prototype masked sw=0.5

Training completed successfully with the requested fixed configuration. Checkpoint: `joint_train_logs/kn1hgnni-rrsd_speaker_svproto_masked_sw05_1000/epoch=0-step=1000.ckpt`. Train loss epoch: 3081.548. Training log: `experiment_logs/rrsd_speaker_svproto_masked_sw05_1000.log`.

The checkpoint was evaluated on the existing 32-utterance, 16-speaker disjoint CV manifest using CUDA, EMA, RRSD PC sampling, N=30, SNR=0.5, seed 2026. Enhanced wavs are in `experiment_logs/rrsd_speaker_eval_svproto_masked_sw05_1000`.

| metric | new mean | delta vs Router (95% CI) | delta vs masked sw=0 (95% CI) |
|---|---:|---:|---:|
| CAM++ enhanced-clean cosine | 0.842476 | +0.001227 [0.000255, 0.002230] | -0.000386 [-0.000778, -0.000007] |
| PESQ | 1.689719 | -0.001804 [-0.006244, 0.001998] | -0.000304 [-0.003550, 0.002298] |
| STOI | 0.797873 | -0.000252 [-0.000484, -0.000050] | +0.000048 [0.000020, 0.000079] |
| SI-SDR (dB) | 6.837560 | -0.128791 [-0.146473, -0.111512] | +0.015217 [0.013158, 0.017334] |

Independent CAM++ same-speaker cross-utterance cosine: 0.840534 (Router 0.842322; masked sw=0 0.840694).

The requested paired bootstrap used 20,000 resamples with seed 2026. No data or prior checkpoints were modified. An initial evaluator invocation failed due to a missing `PYTHONPATH` when launched from `/tmp`; it was corrected and rerun successfully.

## DNSMOS follow-up

The new 32-file output was evaluated with the mymetrics-compatible TorchMetrics DNSMOS evaluator. Means were P808 **3.921740**, SIG **3.428758**, BAK **4.117883**, and OVRL **3.111421**. Per-utterance values are in `experiment_logs/rrsd_speaker_eval_svproto_masked_sw05_1000/dnsmos_metrics.csv`, with the run log beside it.

A paired DNSMOS delta/CI against Router and masked sw=0 is unavailable: `experiments/rrsd_cv_repeat32/router` and `masked_pair32` contain no DNSMOS outputs, while the existing speaker DNSMOS CSV uses a different manifest and filenames. No additional inference was generated.
