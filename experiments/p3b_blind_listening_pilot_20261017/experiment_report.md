# P3b blinded listening pilot package

## Status

Completed exploratory single-listener pilot. All 24 trials were rated tie/no discernible preference; this is not evidence of perceptual equivalence or population-level preference.

## Question and design

Can a listener consistently distinguish the existing Router, `t_cut=0.05`, and selective-window outputs on matched utterances, and which version sounds more natural / less distorted? The pilot uses 8 utterances selected reproducibly from four nominal-SNR strata (2 each) of the existing 32-utterance development manifest. For each utterance, all three condition pairs are presented once, giving 24 trials. Side assignment and trial order are randomized with seed `20261017`.

## Matched inputs and provenance

The three existing output directories contain the same 32 manifest filenames. Their inference setup is documented as the same 80k EMA checkpoint, per-file seed 2026, PC sampler, `N=30`, `snr=0.5`, no corrector. Conditions are Router scale 1, `t_cut=0.05` scale 1, and selective scale 1 over `0.15 <= t <= 0.75`. The Router and t-cut commands are recorded in `experiment_logs/rrsd_tcut_screen/*.log`; selective setup is in `experiments/rrsd_selective_screen/config.yaml` and its report.

Only existing generated WAVs were copied into `playback/audio/` under anonymous trial/side filenames. No dataset file, checkpoint, model, or inference output was modified; no inference, training, or GPU work was run. The answer key and source checksums are in `scoring/answer_key.csv`; do not inspect it until ratings are complete. `playback/index.html` contains the blind player, and `responses.csv` is the response template.

## Rating procedure

Use headphones in a quiet setting. Fix playback volume before starting and leave it unchanged; do not loudness-normalize the files. For each trial, choose A, B, or tie/no preference based on overall naturalness and distortion, optionally noting residual noise, metallic/robotic sound, discontinuities, timbre, or intelligibility. “Strength 1” means a slight preference and “2” a clear preference. If uncertain, use tie. Do not guess the system identity.

## Limitations and interpretation

This is one listener and eight utterances from a repeatedly used development set. Utterances are reused across three comparisons, so trials are not independent. Results can identify obvious, repeatable listening issues and guide the next experiment, but cannot establish general human preference, independent generalization, statistical significance, or paper-level perceptual improvement. If this pilot is informative, repeat with independent listeners and a newly locked set before selecting a training target.

## Current artifacts

- `config.yaml`: fixed selection, pair randomization, and source conditions.
- `metrics.json`: pending outcome schema; contains no measured ratings.
- `playback/index.html` and `playback/audio/`: anonymized listening material.
- `responses.csv`: fill-in response sheet.
- `scoring/answer_key.csv`: condition/file mapping and source hashes; open only after rating.
- `prepare_blind_ab.py`: reproducible package-generation script.


## Listener outcome (2026-10-17)

The listener rated all 24 comparisons as `tie`: 8 Router vs. `t_cut=0.05`, 8 Router vs. selective, and 8 `t_cut=0.05` vs. selective. There were no A or B preferences and no strength ratings. The listener reported that matching regions were similarly clear or unclear across versions and that the differences were not discernible.

This is a negative, exploratory result: under this listener and playback setup, these three existing routing settings did not yield an audible preference on the selected eight development utterances. It does **not** prove that the systems are perceptually identical, that no other listener would distinguish them, or that differences cannot appear on other utterances. Trials share utterances and come from a development set already used for route selection, so no inferential significance or generalization claim is made.

**Decision:** the pilot provides no specific repeatable audible defect or preferred arm to supervise a perceptual loss or conditional discriminator. Do not start P5a/P5b from this result or describe small automatic-metric shifts as demonstrated listening improvements. Reconsider only if independent listeners on a newly locked sample identify a consistent preference or specific distortion.
