# P8 frontend/backend audit audio export

## Manifest

Eight CV utterances were selected read-only from `wav.scp`, one per speaker, with `_snr-(\d+)` parsed as negative SNR. All selected SNRs are -16 to -19 dB. Speakers in `rrsd_cv_repeat32` and `tr0_fresh_cv_manifest_20261018` were excluded. Inputs are symlinks only.

## First failure and repair

The initial Y_pre attempt failed before model loading because PyTorch 2.6 defaulted to `weights_only=True`; the unsupported global was `JointWaveScpDataModule`. This is recorded in `logs/y_pre.log`. The minimal repair added `weights_only=False` to `joint_ablation_by_dir.py`; `py_compile` passed. The original failed log was preserved.

## Retry results

After the repair, all three requested arms completed on CUDA device 0 with the locked EMA checkpoint, PC sampler N=30, corrector none, SNR 0.5, seed 2026:

- Y_pre (`frontend_only`): 8/8, 14 s.
- Router-off (no `--rrsd`): 8/8, 87 s.
- Router (`--rrsd --rrsd_scale 1.0`): 8/8, 88 s.

All outputs were readable 16 kHz WAVs with 480000 samples (30 seconds). Per-file SHA256, byte counts, and timing records are in `metrics.json`; retry command logs are `logs/y_pre_retry.log`, `logs/off_retry.log`, and `logs/router_retry.log`.

## Listening material

`listening_pack/` contains 24 labeled 12-second clips (8 utterances × 3 arms). Each arm uses the same centered crop from its 30-second source. `LISTENING_GUIDE_CN.md` gives Chinese listening instructions; `key.csv` records the arm mapping and clip hashes. This is a qualitative artifact screen only: the dataset's SCP `text` entries are `dummy`, so no human WER/CER or transcript claim is made here. No quality metrics were run. Dataset and checkpoint were not modified.

## Auxiliary objective metrics

After the audio export, PESQ/STOI/SI-SDR and DNSMOS were computed from the same 8 source pairs. These are auxiliary and do not establish content preservation. Means (`n=8`) were:

| arm | PESQ | STOI | SI-SDR (dB) | DNSMOS-OVRL |
|---|---:|---:|---:|---:|
| Y_pre | 1.2049 | 0.6811 | 2.1413 | 2.5277 |
| Router-off | 1.2474 | 0.6804 | 0.9793 | 2.8893 |
| Router | 1.2469 | 0.6823 | 1.6286 | 2.8398 |

Paired Router minus Router-off: PESQ `-0.0004`, STOI `+0.0019`, SI-SDR `+0.6493 dB`, DNSMOS-OVRL `-0.0495`. The evaluator requested automatic CUDA selection, but ONNX Runtime did not expose `CUDAExecutionProvider`; DNSMOS therefore ran on CPU. Per-file CSVs and `metrics/summary.json` preserve the raw results. No significance claim is made from eight utterances.
