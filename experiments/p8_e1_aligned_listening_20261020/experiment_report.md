# Experiment report: P8-E1 aligned listening repair

## Scope

This is a read-only diagnostic/listening package. No model inference, training, DNSMOS, checkpoint modification, or overwrite of source/old listening files was performed. New artifacts are under this directory only.

## Inputs and selection

The eight source utterances and generated arms are listed in `../p8_frontend_backend_audit_20261020/manifest.csv`. Four unused central-12-second windows were selected from `../p8_e1_candidate_ranking_20261018/candidate_windows.csv`: fileid 4812 at 17.0–17.5 s and 14.5–15.0 s, fileid 10720 at 9.0–9.5 s, and fileid 6339 at 10.5–11.0 s. Prior counterexample-pack windows were excluded. Each output clip spans target ±0.75 s and is PCM-16 WAV at 16 kHz.

## Alignment evidence

All 20 source arms were checked at 16 kHz and 480000 samples (30.0 s). For each selected context, a normalized cross-correlation scan against clean over ±1600 samples (±100 ms) found best lag **0 samples / 0 ms for noisy, clean, Y_pre, Router-off, and Router**. Source and output SHA-256 values are recorded in `metrics.json`. This rules out a detectable file/time offset in the inspected range and crop construction.

## Interpretation boundary

Noisy-clean local zero-lag correlations are 0.463, 0.391, 0.158, and 0.097; candidate clean-aware local SNRs are −10.86, −11.38, −14.28, and −17.12 dB. Thus noisy may fail to make the sentence/phonetic content identifiable under masking even while the speaker remains recognizable. This is a weak-evidence/noisy-observability explanation, not evidence of file misalignment.

If a listener finds Y_pre, Router-off, or Router content different from clean in these synchronized clips, label it as **model semantic deviation/content distortion** pending independent transcription. Do not report it as a file offset: the alignment scan found zero lag across all arms. Conversely, this package does not prove that any particular syllable is unrecoverable; it only improves temporal comparability and exposes the distinction.

## Reproducibility

`build_pack.py` contains the fixed candidate list, crop rule, source resolution, hash calculation, and lag scan. `config.yaml`, `metrics.json`, and `alignment_scan.json` provide the machine-readable record; `README_CN.md` and `listening.html` provide the listening instructions.
