# I0 software preparation — NOT EXECUTED

Only this directory was written. Root analysis_plan.md/config.yaml unchanged. No real utterance selection/registration, no real WAV/header inspection, no PCM sample decoding, no model/audio-library imports, no network/inference/training. Only manifest first-row schema was read; no actual source-row selection ran. Synthetic tests use temporary JSON/PCM-header containers under this directory and fake paths. Whole-file hashing is explicitly different from PCM decoding: future24WAV SHA reads payload bytes, while sample decoding remains0; runtime reports wav_header_checks and full_wav_hash_bytes separately.

Implemented only I0: exact6lexical speakers×2numeric-ID utterances, localtrain3–5s inclusive and suffix>=26, six excluded speakers. Duplicate/bad eligibility schema/header mismatch/shortage fail without fallback. Driver+independent validator rederive source ordering separately and use raw binary JSONL-line SHA (including CRLF). Actual selected paths must resolve inside fixed /data/datasets/vb/voicebank_demand/16k/train/{clean,noisy}/ with matching ID, rejecting parent-symlink escape. Headers only getparams; declared payload truncation is rejected without reading PCM samples. No alignment/gain/residual/editing/detector/CAEG implementation.

Provenance scan is strictly maxdepth2/100candidates/512KiB under configuredroot, only allowedtext suffixes, test/valid subtrees and symlinks excluded. Saves bounded unreviewed clues; never infers original noise recording from a filename, category or algebra. Identity remainsunknown, noise_disjoint_ready=false/split_frozen=false/scientific_gate_ready=false even after metadata success. Any future evidence-based identity assignment requires separate explicit review, not an automatic fallback.

Single process/no subprocess or retry. Exclusive STARTED;58s workload alarm then absolute60s hard exit124 covering source postchecks and report writes. Hard timeout records hard_timeout_failure.json and independent success validator rejects it. Two-MiB total including preparation with reserve checks; finalelapsed and byte cap enforced. Guard refuses sourcewrites, unapproved dataset reads, model/audio imports, sockets/processes. Config current_launch_authorized=false stays unchanged: commands below require separate Root review and Worker authorization, not permission from this report.

Future Worker registration, ONCE after Root authorization:
```
python3 -B /data/coding/sgmse/experiments/p8_noisy_local_identifiability_prep_20261019/register_metadata.py --execute-i0-after-root-review
```
After completed registration, independent standard-library validator:
```
python3 -B /data/coding/sgmse/experiments/p8_noisy_local_identifiability_prep_20261019/validate_metadata.py
```
Runtime standardized config.yaml/metrics.json/experiment_report.md and registry.json are absent now. Success only establishes traceable metadata, not I1 alignment, noise purity, phoneme identity or scientific identifiability. Failures retained and no automatic retries. Future validator rehashes only24 selected WAVs (plus source JSONL metadata/limited scanned text), not whole corpus.
