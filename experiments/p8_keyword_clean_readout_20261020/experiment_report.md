# Fixed64 Clean keyword readout: acquisition timeout; applicability unknown

The unique authorized command exited1, PID5949/session33664 (completed), after 120.018421s. The120s download alarm interrupted a response read; no retry, resume, mirror, deadline extension, package install or prompt/model rescue followed. This is a technical acquisition failure, not a failed Clean recognition score.

```text
env/bin/python -B experiments/p8_keyword_clean_readout_20261020/readout.py --execute-once
```

One HTTPS GET was issued to [the official mini Speech Commands archive](https://storage.googleapis.com/download.tensorflow.org/data/mini_speech_commands.zip). HTTP200, Content-Length182082353 and ETag"4b8a67bae2973844e84fa7ac988d1a44" matched the frozen source. Only11534336 bytes in fully returned read chunks were saved, so the archive is incomplete. The byte counter measures returned/written chunks; socket buffers or partial data not returned by the interrupted read are not independently metered as physical network bytes.

Partial filename mini_speech_commands.zip (preserved; not a valid completed archive), size11534336 bytes, SHA256 `039c508770e1f258d5b1118db9fd195158be0853464090a9fa0fe3724462871d`. A partial full-archive response can contain arbitrary members' compressed bytes, including reserved groups; this is not claimed to avoid reserved compressed data. No ZIP member, reserved or selected, was parsed/extracted/decoded.

Actual activity: archive GET1; member read attempts0; extracted members0; selected-audio bytes0; header checks0; ASR model loads0; transcribe calls0; internal decode calls0; frontend0; backend0; optimizer0. No audio or StoRM/CPEL model was accessed. The local tiny.en weight was byte-hashed before acquisition and after failure, not loaded. All14 installed Whisper Python source hashes were checked and remain unchanged.

Observed item count0 means correctness, per-word counts, empty/multiword/OOV flags and the58/64-plus6/8-each applicability gate are unmeasured/unknown. Neither missing results nor the technical failure are counted as64 recognition errors. The independently saved scalar activity check reconstructs only this bookkeeping from metrics.json; it performs no ASR/audio recomputation. No prompt/word-list input, ASR replacement, classifier training or subsequent enhancement run was attempted.

All30 selected source/code/config/manifest/authorization/ASR-package and weight/goal/old-source-failure/CPEL result hashes match pre/post. Original source-feasibility network/artifact failures remain unchanged. This approved run was a separate acquisition/readout question, not a repair or reclassification of that source result. Frozen manifest64 and testing reservations remain intact. No unrelated704-WAV hash scan or CPEL terminal weight read occurred.

License attribution remains CC BY from official TensorFlow documentation supplied by Root; archive README/license, labels against recordings, true speaker identities and pretraining exposure remain unverified. Prefix is a filename group only. This failed acquisition supplies no enhancement, phoneme/content or natural noisy-domain reliability evidence.

Runtime artifact count reported11601794 bytes and disk free1604825088 bytes, within200MiB and1GiB reserve. Final report/receipt sizes are recorded separately in execution_receipt.json. Primary metrics SHA256 `3323188a5f8a3fca87258581e233c2855d07b9b629d22997085e9963d6105bb8` remains unmodified. Exact HTTP headers, partial progress and complete timeout traceback are in metrics.json. Root attachment, current code/config, local ASR and previous result preservation hashes plus actual command/PID/session are in execution_receipt.json.

Bridge39452 and listening server16022 remain alive and untouched. The unique run is stopped. There is no automatic new data-acquisition strategy, full download, Clean readout, noisy/frontend/backend experiment or training authorization following this failure.
