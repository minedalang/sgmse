# Fixed64 Clean keyword readout, transport02: completed; applicability failed

Unique child execution completed with exit0, PID6206/session94778 (completed). Total104.054191s; acquisition94.842869s. All64 frozen member acquisitions/CRC/header checks succeeded and64 fixed tiny.en transcription calls ran, but the preregistered Clean applicability gate FAILED:49/64 correct (76.5625%) versus≥58/64, with down4/8 and no5/8 below the≥6/8 per-word minimum. This is a completed measurement with a negative applicability result, not a technical failure or an enhancement experiment.

```text
env/bin/python -B experiments/p8_keyword_clean_readout_20261020/transport_attempt02/readout.py --execute-once
```

Selection/audio/model/options/scoring/screen are identical to the parent64 manifest. Only the independently authorized transport changed. The parent120s full-download failure and partial file are unchanged. This child did not resume that partial, change members, or choose easier files from ASR output.

| Word | Correct / fixed total | Gate minimum |
|---|---:|---:|
| down | 4/8 | 6/8 |
| go | 7/8 | 6/8 |
| left | 7/8 | 6/8 |
| no | 5/8 | 6/8 |
| right | 6/8 | 6/8 |
| stop | 6/8 | 6/8 |
| up | 7/8 | 6/8 |
| yes | 7/8 | 6/8 |

Output flags (nonexclusive): empty=1, multiword=1, out-of-vocabulary=12; single-token wrong in-vocabulary=2. All64 original texts/tokens/labels/groups are retained without homophone/spelling/dictionary replacement. These flags describe recognizer output, not verified human content errors.

Incorrect recognizer records:

| Index | Label | Group | Raw text | Normalized tokens |
|---|---|---|---|---|
| 3 | no | bdee441c | " Now..." | now |
| 8 | down | dbb40d24 | " Yeah." | yeah |
| 9 | go | dbb40d24 | " Cool." | cool |
| 10 | left | dbb40d24 | " Now." | now |
| 11 | no | dbb40d24 | " We know." | we know |
| 12 | right | dbb40d24 | " Okay." | okay |
| 13 | stop | dbb40d24 | " Go." | go |
| 14 | up | dbb40d24 | " Um." | um |
| 15 | yes | dbb40d24 | " Okay." | okay |
| 19 | no | c6ee87a7 | " Now." | now |
| 20 | right | c6ee87a7 | " Great." | great |
| 24 | down | 099d52ad | " Don." | don |
| 37 | stop | 7fd25f7c | " ." |  |
| 40 | down | 439c84f4 | " Darling." | darling |
| 48 | down | 7c1d8533 | " No." | no |

Acquisition used64 official HTTPS GET ranges, each206 with exact Content-Range/Content-Length and frozen ETag; actual response bodies total5535648 bytes (≤8MiB), all64 range files complete. The known source archive length is182082353, but no complete archive was downloaded and no full-archive SHA is available. Overfetched neighboring compressed bytes may include reserved data; no neighboring/reserved member was parsed, extracted or decoded. Only the64 selected member raw-deflate segments were bounded-inflated and CRC checked. All64 original WAVs are mono PCM16/16k with1–16000 samples; no resampling, cropping, external padding or audio regeneration. Selected raw bytes total2050816. Per-range actual headers/bytes/timing/body SHA/compressed SHA/raw SHA and bit3 local-header checks are preserved in metrics.json.

ASR loaded once from the prehashed local tiny.en weight. English, temperature0, beam1, no initial prompt/word list, no previous context, no_speech_threshold=None, fp16=False were fixed; effective decoding options and64 internal decode calls are recorded. One transcribe API call is not claimed to equal one neural forward. No source ASR fetch/install or network after acquisition occurred. Labels were used only for scoring, not model input. No StoRM/frontend/backend, noise generation, CPEL terminal weight read or training occurred.

Independent saved-JSON arithmetic re-tokenized all64 raw texts, checked exact-label and empty/multiword/OOV flags, per-word and total counts, applicability Boolean, all64 exact HTTP ranges/body-size sums, raw-member SHA=recorded WAV SHA, and64 internal decode-call sums. It performs no model/audio recomputation. These hashes are saved-output consistency checks, not a second PCM/model validation.

All34 protected current config/code/manifest/receipt/source metadata/local ASR weight and14 package sources/goal attachment/old CPEL metrics/parent failure artifacts match pre/post SHA. Parent partial SHA remains fixed; old source-feasibility timeout/artifact-cap failure is not reclassified. Bridge39452 and server16022 remain alive and untouched. Full preservation hashes, actual command/PID/session/terminal status, final artifact/disk counts are in execution_receipt.json. Primary metrics SHA256 `2113502e6ae2523b673a731fda001ad90bef9d789cd07904b5cdb3479136322d` was not modified for reporting.

This Clean readout does not meet the fixed threshold for adopting it as a content-transfer endpoint. No prompt/model/classifier rescue or subsequent noisy/frontend/nativeOFF experiment was started. Prefix is not a verified speaker identity; public-data pretraining exposure and recording-label accuracy against human judgments remain unknown. CC BY attribution comes from official TensorFlow documentation; archive README/license was not inspected. This single64 validation screen is not a general ASR or human intelligibility estimate.

The transport child is complete and stopped. Parent failures and the closed CPEL v2 configuration remain intact; no third acquisition strategy or automatic follow-up is authorized.

Root decision: close this fixed tiny.en keyword-identity readout branch. No Noisy/frontend/nativeOFF evaluation, prompt/model change, KWS training or testing-set sample rescue follows. This result does not prove that the source word labels are wrong or that physical speech content is unidentifiable.
