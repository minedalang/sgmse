# Rollout-aware adapter independent replication (2026-10-18)

## Protocol

This is an independent speaker-disjoint development replication on 4 speakers using the first 8 rows of the locked `tr0_fresh_cv_manifest_20261018/manifest.csv`, representing 4 speakers not used by the earlier rollout pilot (2 utterances per speaker). Six utterances from 3 speakers were used to fit the adapter; two utterances from the fourth, unseen speaker were held out. The official test set and source dataset were not modified.

The module-33 adapter was zero initialized and was the only trainable component. Training used Adam, `lr=1e-5`, 100 steps, gradient clipping 1.0. Rollout states used PC reverse diffusion (`N=30`, `snr=0.5`, no corrector), with trace steps `[3,5,7,9]`. The cache is approximately 945 MiB.

## Proxy result

| split | baseline NMSE | adapter NMSE | relative change |
|---|---:|---:|---:|
| train utterances (6) | 1.022171 | 0.442414 | −56.72% |
| unseen-speaker holdout utterances (2) | 0.595017 | 0.268770 | −54.83% |

The holdout proxy improvement reproduces the direction of the earlier pilot, but two utterances are insufficient for a generalization claim.

## Audio metrics

The aggregate eight-utterance values include six adapter-training utterances and two holdouts, so they are descriptive rather than independent evaluation:

| metric | baseline | adapter | delta |
|---|---:|---:|---:|
| PESQ WB | 1.73847 | 1.77811 | +0.03964 |
| STOI | 0.839823 | 0.839896 | +0.000073 |
| SI-SDR (dB) | 7.07699 | 7.24361 | +0.16663 |
| DNSMOS OVRL | 3.21732 | 3.22629 | +0.00897 |

On the two unseen-speaker holdout utterances, paired deltas were PESQ `+0.00157`, STOI `+0.00153`, SI-SDR `+0.164 dB`, DNSMOS-P808 `+0.00750`, but DNSMOS-SIG `−0.03373`, DNSMOS-BAK `−0.00539`, and DNSMOS-OVRL `−0.03465`. This is mixed perceptual evidence and cannot establish a quality gain.

Per-utterance metrics and the train/holdout split summary are in `per_utterance_metrics.json` and `audio_metric_split_summary.json`. DNSMOS emitted an ONNX CUDA-provider warning; provider execution was not verified, so these DNSMOS values may have used CPU fallback.

## Decision

The independent replication supports the narrow statement that rollout-state adapter training can reduce the clean-x0 proxy on a small unseen-speaker holdout. It does not yet establish improved perceptual quality, Corrective Forcing, or a paper innovation. The next gate is a larger independent CV/holdout with a predeclared metric rule; do not use the frozen official test set or start long training based on this result alone.
