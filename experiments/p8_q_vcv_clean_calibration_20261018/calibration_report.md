# Clean VCV diagnostic calibration

Status: prepared; inference has not been run by the engineer.

The fixed protocol is `calibration_protocol.yaml`. The Research Lead reviews this design before the Experiment Worker executes `calibrate_clean.py`. The worker must record the approval reference with `--design-approved-by`.

## Design

Primary analysis uses 16 corpus nominally labelled clean clips: four speakers, k/g and t/d, two identities per contrast, with vowels and stress matched within each pair. Up to 13 supplementary manual clips remain descriptive. Labels, pair selection and audio hashes are frozen before model output. Entire VCV waveforms are scored; manual boundaries do not select scoring regions.

All 24 fixed consonant candidates receive exact three-token VCV CTC forward log likelihoods. Vowels are fixed to a→æ, i→iː, u→uː. Binary margins use k versus g or t versus d. No mapping adjustment or sample selection follows results. Greedy IPA runs remain descriptive.

## Results — worker/research lead to complete

- Frozen manifest SHA256: pending
- Model/source/script/protocol hashes: pending (`calibration_outputs/frozen_inputs.json`)
- Environment and execution status: pending
- Primary 24-way correct /16: pending
- Primary binary correct /16: pending
- k/g 24-way correct /8: pending
- t/d 24-way correct /8: pending
- Speaker, contrast and speaker/contrast metrics: pending
- Confusion matrix and individual margins: pending
- Supplementary descriptive results: pending
- Gate decision: pending

The preset gate requires at least 12/16 primary 24-way, 14/16 primary binary, and 6/8 24-way for each contrast. Any failure stops the noisy/enhancement stage. No confidence intervals or independence assumptions are made.

## Reproducibility and limits

Run once into a newly created `calibration_outputs` directory. Preserve completed and failed outputs; do not retry without diagnosis and Research Lead direction. Original 25 kHz WAVs remain unchanged; 16 kHz resampling occurs in memory. Saved float32 logits and probabilities allow offline score reuse. Total experiment artifacts are capped at 25 MB. Source code, checkpoints and existing datasets are unchanged.

A pass establishes diagnostic calibration within this controlled clean corpus only. It does not establish human recoverability, noisy discrimination or generalization. The Research Lead merges this report with retrieval results into the final experiment report and records the next scientific decision.
