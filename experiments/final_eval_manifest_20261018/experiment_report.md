# Official DNS Ins20 final evaluation manifest audit

## Status

Read-only candidate lock completed. No audio was written, no dataset file was changed, and no checkpoint was modified. No enhancement or metric inference was run.

## Evidence

- Source: official test_set/synthetic/no_reverb and test_set/synthetic/with_reverb.
- 150 utterances per split, 300 paired noisy/clean rows total.
- All files exist, are 16 kHz, and noisy/clean sample counts match.
- The manifest was compared against all five existing experiment manifests found under experiment_logs/**/manifest.csv and experiments/**/manifest.csv.
- Canonical path overlap was zero. Some file IDs/basenames overlap because the dataset uses separate directory trees; those are not evidence of reuse.
- Manifest SHA256: 4c91782fb1459bd14f2b8581b1f4b537ca98bafd907ecc961226883213bcfa2b.

## Separation rule

The previously screened 150-item manifests under synthetic_snr-20_-5_testset and the P7/CV manifests are not this official test set. This lock must remain untouched until model, routing rule, checkpoint, sampler, evaluator, and statistical protocol are frozen.

## Files

- manifest.csv: paired paths and per-file SHA256 values.
- config.yaml: source, hashes, and validation protocol.
- metrics.json: audit summary.
