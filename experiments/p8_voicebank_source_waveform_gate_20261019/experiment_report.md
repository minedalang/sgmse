Official/local VoiceBank four-file waveform gate: waveform_gate_failed

The frozen zero-delay full-utterance conversion-consistency gate was reached and failed on all four files. Official HTTP transport, archive-member CRCs, selected-file lengths and evidence serialization succeeded. This is a numerical inconsistency between the single frozen default float32 CPU 48k→16k recipe and the existing local 16k clean files. It does not establish wrong audio identity, wrong transcript, or the cause of the historical conversion difference. High correlation indicates strong zero-delay waveform similarity; it does not override the preregistered maximum-error criterion.

| ID | Local/derived samples | Maximum error (PCM16 LSB) | RMS error (LSB) | Correlation | Local-to-difference ratio (dB) | Gate ≤2 LSB |
|---|---:|---:|---:|---:|---:|---|
| p226_027 | 63447 | 262.085083008 | 14.874868531 | 0.999968128180 | 41.939209452 | Fail |
| p226_028 | 67653 | 572.694702148 | 34.725000535 | 0.999786666218 | 33.696685405 | Fail |
| p287_037 | 60008 | 432.624511719 | 21.581062923 | 0.999903716031 | 37.153044408 | Fail |
| p287_054 | 51089 | 431.011901855 | 28.682933808 | 0.999903741767 | 37.152427220 | Fail |

The field `snr_db` in the original metrics is 10*log10(mean(local_clean²)/mean((derived−local_clean)²)), ranging 33.70–41.94 dB. It is the local waveform to conversion-difference ratio, not original clean/noisy mixture SNR. All comparisons use complete utterances at zero delay, with exact length matching. Raw float equality and serialized derived PCM16 equality with local are false for all four files. No gain, lag, trim, threshold or resampler search was performed.

One frozen execution started 2026-10-19T03:06:09.186048+00:00: session 75430, supervisor PID 43408, child PID 43411; elapsed 14.611230138 seconds, exit code 1 reflecting gate failure. Exact command and provider verification are recorded in execution_provenance.json. No rerun or retry was issued.

All 15 expected HTTP requests returned 206 with exact Content-Range and total archive size 2,486,057,279, identity encoding and stable nonblank ETag/Last-Modified. ETag was "d2d5a45ec32f8fcbf201bde0447e20ba", Last-Modified Mon, 21 Aug 2017 10:32:37 GMT. Body bytes total 2,286,268; conservative network accounting is 3,269,308 bytes including 65,536 reserved header bytes per response, below 8,000,000. No full archive was downloaded; its published MD5 remains metadata only, not a full-archive verification. Four selected official ZIP members passed size/CRC checks. Original response headers, their hashes, request ranges, payload bodies/hashes and frozen selected central records remain preserved.

The single default torchaudio.transforms.Resample(48000,16000) instance used float32 CPU, width 6, rolloff 0.99, sinc_interp_hann, beta null. Four resampling calls and eight source/local decodes were performed. Versions: Python 3.10.22, NumPy 2.2.6, SoundFile 0.14.0, Torch 2.14.1+cu130, torchaudio 2.11.0+cu130. Installed defaults matched the frozen recipe. Preparation-script presence does not prove historical preparation identity.

Independent offline verification (`verify_saved_evidence.py`, `independent_validation.json`) passed in 0.113497934 seconds. It imports no gate implementation, blocks sockets, has a 60-second alarm and uses one numerical thread. It checked 13 preparation locks, 10 source locks against pre/post/current hashes, all 47 original run artifacts, all 15 HTTP ranges/status/headers/body hashes, stable version identifiers, ZIP directory and selected local/member bytes/CRC, and all four official WAV headers. Saved float32 arrays were compared again to original local PCM, reproducing maximum/RMS errors, correlations, difference ratios and failed gate decisions. Each serialized derived WAV exactly matches the independently computed floor/saturate PCM16 samples. Verification used 0 network requests, 0 resampling calls, 0 models and 0 training. Original metrics.json and its artifact manifest were not edited; its artifact byte count is the pre-final snapshot, not the completed directory size.

Original local data, source files, weights, dirty worktree, listening server PID 16022 and compatibility bridge PID 39452 were preserved. Git status before/after the gate was identical. Provider was read-only verified as codex_tools_compat. No test content, ASR, enhancement, training or model was opened or run. No automatic retries, fallback downloads or post-failure repair occurred.

Selection remains the pre-audio metadata rule for p226/p287 valid: first two lexicographic IDs with numeric utterance≥26 and duration 3–8 seconds. This result does not establish whole-dataset spoken alignment, phoneme timing, transcript error, pretraining holdout or enhancement performance. The gate remains failed at its frozen threshold. Any later use of saved official derived audio as a new explicit reference requires a separate Research Lead design and cannot reclassify this gate. No pilot or listening package was created.

Final artifact bytes including this report, execution provenance and offline validation: 5522248, below 12,000,000.
