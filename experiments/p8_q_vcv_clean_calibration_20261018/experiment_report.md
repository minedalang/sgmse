# VCV subset retrieval and clean calibration (2026-10-18)

## Completed result

The subset was frozen before recognition and retrieved successfully. The
frozen wav2vec2 recognizer passed the preset clean calibration gate for the
16 primary clips. The extra manual-boundary clips show substantial label
disagreement (7/12 agree), limiting generalization. No noise experiment,
speech-enhancement inference, or training has been executed.

| Set | 24-way nominal identity agreement | k/g or t/d forced-choice agreement |
|---|---:|---:|
| Frozen primary, 4 speakers | 16/16 | 16/16 |
| k/g primary subset | 8/8 | 8/8 |
| t/d primary subset | 8/8 | 8/8 |
| Supplementary manual-boundary recordings | 7/12 | 7/12 |

Primary speakers are f6, m2, f11, m4. Every primary pair matches speaker,
aCa vowel context and front stress (1). Samples were selected deterministically
from archive availability, before listening-model outputs; none was discarded
because of recognition performance. Supplementary recordings do not enter
the gate. Disagreement with nominal labels does not by itself distinguish a
pronunciation deviation from a recognizer error.

## Method, costs and evidence

HTTP ranges retrieved only selected members of the official train.zip:
1,637,934 body bytes across 115 HTTP206 requests, no proxy fallback. The
full 169,755,879-byte archive was not downloaded. The 28 unchanged mono25kHz
WAVs occupy 691,788 bytes; CRC32, SHA256, pair matching and frozen-manifest
checks pass. The complete archive MD5 is a metadata reference, not a
download verification. Corpus provenance and license identifier CC BY4.0
are retained from the official record.

Calibration resamples WAVs in memory using polyphase25kHz-to16kHz conversion
and uses the existing local `facebook/wav2vec2-lv-60-espeak-cv-ft` model.
For each waveform the scorer evaluates all24 exact three-token VCV sequences
using a fixed vowel map (a=æ,i=iː,u=uː) and corpus-to-IPA consonant map;
the ranking is independent of the nominal target supplied for evaluation.
Binary scores compare only the predefined k/g or t/d alternatives. The
primary mean log-likelihood margin against the paired competitor is 3.758226.

The design was frozen in calibration_protocol.yaml: at least12/16 primary
24-way,14/16 binary, and6/8 24-way for each contrast. All passed. These
small, dependent, selected corpus examples do not establish a population
accuracy estimate or human intelligibility; no confidence interval is claimed.
Free greedy IPA is descriptive and is not substituted for the chosen metric.

Process wall time was8.99s; the script's full load/hash/score/write runtime
was3.527720s. Pure CUDA forward time was not separately instrumented.
The initial `/usr/bin/time` launcher was unavailable (exit127 before Python
or outputs); the actual calibration ran once with Bash builtin timing.
Offline verification recomputes all28×24 saved CTC likelihoods exactly from
saved logits, confirms predictions/gate and checks source hashes unchanged.
Whole-experiment storage after calibration was about6.62MB.

## Decision and next step

Use the original frozen16 clean cases as a controlled positive-control
candidate for the next fixed-noise diagnostic. Do not broaden the claim to
other contexts, DNS sentences, or noisy recovery based on the primary clean
score. The12 supplementary cases must remain visible; do not tune vowel
mapping or replace samples after seeing their outputs. Next freeze noises,
SNRs and random seeds, evaluate speech-identity sensitivity and noise
stability before deciding whether any StoRM experiment is informative.
This result alone does not justify content-loss or architecture training.

## Authoritative artifacts and reproduction

- config.yaml: immutable subset retrieval preparation protocol.
- calibration_protocol.yaml: immutable calibration design snapshot.
- preparation_metrics.json: preserved original preparation status.
- metrics.json: consolidated completed-stage status and hashes.
- retrieval/: archive inventory, pre-audio selection plan, frozen manifest,
  raw WAVs, stage metrics and worker_validation.json.
- calibration_outputs/: per-sample24-way scores, logits/probabilities,
  metrics.json, worker_validate.py and worker_validation.json.

```bash
env/bin/python experiments/p8_q_vcv_clean_calibration_20261018/fetch_subset.py
env/bin/python experiments/p8_q_vcv_clean_calibration_20261018/calibrate_clean.py \
  --device cuda --design-approved-by 'Research Lead fixed clean calibration design 2026-10-18'
```

Both stages refuse existing output directories. These commands document
the completed run; do not rerun over current artifacts. The calibration
argument records scientific design provenance and does not request a new
user approval. Retrieval has a32MB output guard and calibration a stricter
25MB whole-experiment guard; measured outputs remain well below both.

## Preserved preparation notes (historical)

Prepared fetch_subset.py and protocol config. No network retrieval, training, or enhancement inference has been performed. Existing external corpus audit metadata and manual MLF provide source provenance.

The script inventories train.zip using bounded HTTP ranges and freezes 8 matched k/g and t/d pairs from 4 corpus train speakers. Both contrasts share vowel context and stress within each speaker. Selection uses filename availability and predetermined ordering. Up to 13 additional manual MLF targets present in the archive are supplementary. Manual segmentation does not establish perfect consonant identity.

Run the script once with a fresh output directory. It refuses full HTTP 200 downloads, records failures, preserves original 25 kHz WAV bytes and verifies WAV CRC32 and SHA256. The complete archive checksum is recorded as metadata and cannot be independently checked without the full archive. The Research Lead supplies the calibration design after reviewing the resulting inventory and frozen manifest.

Offline verification passed syntax compilation, deterministic 4-speaker/8-pair selection, actual zipfile member reads through a local range fixture, rejection of HTTP 200 before any body read, and the single global proxy fallback limit. No remote request was made. Use the repository env/bin/python interpreter; the system Python lacks requests.

The retrieval subdirectory contains selection_plan_frozen.json sealed before audio reads, then selected_manifest_frozen.json (files, pairs, provenance) and selected_manifest_frozen.csv sealed after WAV validation and before any model results. Standard fields are sample_id, absolute wav_path, split=corpus_train, speaker, gender, vowel_left/right, stress, consonant, primary_pair_ids, is_primary, nullable manual_start_s/end_s, boundary_source, and sha256. CSV pair IDs are separated by a vertical bar. JSON retains full manual segments and archive member metadata.
