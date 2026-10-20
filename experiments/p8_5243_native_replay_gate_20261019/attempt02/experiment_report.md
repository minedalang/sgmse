# Native5243 replay gate — attempt02 passed

Passed in 22.32094seconds with exactly1frontend and1OFF backend call,NFE30 and30trace callbacks. This was the sole successful scientific execution. Parent attempt01 remains a preserved import failure with0model calls (6.96014seconds). Across both authorized attempts:1frontend+1backend; total supervisor time29.28108seconds. No intervention,Router,classifier,training,new variants,downloads or installation.

Attempt02 changes only PATH by prepending the existing env/bin, making installed Ninja discoverable. The original five prepared files and frozen_manifest.json were copied byte-identically into this isolated subdirectory. attempt_provenance.json records the correction and hashes of six parent failure artifacts; final independent validation confirms all six remain unchanged. Parent STARTED/results/reports were not overwritten.

|Gate|Historical samples|Bytes|PCM samples|
|---|---:|---|---|
|Recomputed frontend reencodedPCM,passed before backend|480000|Exact match|Exact match|
|NativeOFF full30s|480000|Exact match|Exact match|
|NativeOFF18.75–20.75s|32000|Exact match|Exact match|

The original00_ basename and full30s frontend context were retained. Base2026 resolves to effective seed369124093. Runtime settings:80k EMA,mode2,PC,reverse_diffusion predictor,none corrector,N30,sampling snr0.5,OFF/no calibrators/no regeneration override. Source noisy normalization reference peak is0.431304931640625. Backend anchor hash equals the recovered matched frontend float hash; condition hash equals original full noisy float waveform. Frontend did not change saved CPU/CUDA RNG states.

Recovered full-precision arrays are new artifacts,not comparisons against unavailable historical raw arrays:

- outputs/frontend_raw.npy:float32,[1,1,480000],array SHA256 `5ec2ac6d2957909da3d6098f673da2f5dbdc19cf4840d0a2a6f72735d6054aa3`.
- outputs/backend_raw.npy:float32,[480000],array SHA256 `f2400f84c3e84c9b5327c8dd929db093a6f4f853199642deb485d2835e30a0a1`.

Backend raw was captured before main clipping/serialization; clipping count is0. FullPCM and context WAVs are under outputs/,alongside14saved RNG arrays around frontend/backend/sampler and first callback. Source/core/checkpoint17locks were actually rehashed before and after the replay and all match. Frozen preparation hashes remain unchanged. Historical literal launch command/code hashes remain unavailable; this is a supported reconstructed invocation with exact historical output serialization.

The first trace has step_index0,t=1.0,and state shape[1,1,256,3776],complex64. Its x_t hash is `40fc60d89c8899f1a12c8a38e04bf6d7e05ff20c7ec345b549f64f2e5adc5ae8`. Sampling code calls the hook after predictor:update; x_t refers to state before predictor,after identity none-corrector,while x_next/x_mean are post-predictor. RNG captured within callback is post-predictor and must not be used as before-sampler RNG. Saved before-sampler states are the separate rng_before_sampler files. Trace state arrays were not retained; only their shape/dtype/hash metadata were captured. This distinction is essential for future matched execution.

Independent CPU validation in validate_artifacts.py passed:all30supervisor-manifest entries,all16non-checkpoint source rehashes,actual pre/post checkpoint lock values,frozen copied files,parent preservation,raw descriptors/finite arrays,frontend in-memory PCM bytes,clipped-backend in-memory serialization,full/clip PCM equality and byte hashes,condition/anchor hashes,normalization peak,effective seed,and14RNG files. See artifact_validation.json. No model calls were made during validation.

This gate recovers a usable raw baseline and validates current tracing/serialization for this native input. It does not demonstrate independently recoverable noisy /d/,Router gain,phoneme correctness or a local intervention result. Any local evidence diagnostic requires Research Lead's separate design and authorization. Existing dirty workspace files,data/checkpoint and listening server were preserved. No root-plan edits,commit or push.
