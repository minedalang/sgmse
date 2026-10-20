# P8 file-id 11999 blind forced-choice pack (2026-10-20)

## Status

Pack generation completed; human assessment is pending. This is a read-only auxiliary check and does not establish E1 or phoneme recovery.

## Fixed protocol

- Source: existing aligned P8 paired WAVs and existing `p8_phoneme_probe_20261020/phoneme_results.json`.
- Selection: five highest-ranked existing `fileid=11999` windows with `candidate_frontend_suppression=true`; no new ranking or inference.
- Conditions: clean reference plus noisy, Y_pre, Router-off, Router candidates. Three blind A/B comparisons per window.
- Random seed: `20261020`; trial order and A/B assignment randomized. Hidden mapping is in `answer_key.json`.
- Output clips are small derived PCM16 files. Source hashes were checked before extraction.

## Current result

No human responses are present. The pack cannot decide whether noisy contains an identifiable target or whether a backend restores it. The pre-registered E1 gate requires at least 3/5 windows to satisfy the three-part listener chain documented in `README_CN.md`.

The existing phoneme result was reused only for the 19.50–20.00 s window; it is a CTC acoustic-model proxy and not phoneme ground truth. W01 sample equality against the corrected aligned clean clip: `false/old clip unavailable`.

No dataset, original WAV, or checkpoint was modified; no training or new inference was run.

## Reproduction

```bash
/data/coding/sgmse/env/bin/python experiments/p8_11999_forced_choice_pack_20261020/build_pack.py
```
