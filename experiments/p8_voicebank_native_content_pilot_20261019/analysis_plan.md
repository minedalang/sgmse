# P8 native matched-content measurement pilot — frozen before noisy outputs

## Question, scientific status and retained units

Measure publisher-reference content differences between actual noisy input, its native frontend estimate and the matching 80k EMA OFF backend under natural matched conditioning. No condition swapping, anchor removal, Router, training or architecture changes. The prior failed historical local-conversion gate remains failed. The completed single-ASR clean calibration (39 words/177 scoring characters, no errors) is only a measurement control, not evidence of low-SNR intelligibility or an algorithmic contribution.

All four already-selected calibration IDs in order: p226_027,p226_028,p287_037,p287_054. All are retained, not selected by enhancement outcomes. Two noise sources and two SNRs give16 cases. These are repeated utterances/two speakers and reused historical noises, NOT16 independent samples, a new confirmation set or a pretraining holdout. No confidence intervals/significance/generalization claim.

## Input recipe (CPU preparation only first)

Clean is the complete stored official-derived16k PCM16 WAV for each ID from `p8_voicebank_source_waveform_gate_20261019/outputs/`; float64 values are exact int16/32768. Never substitute old localVB clean or pair these with old localVB noisy. No resampling, trim, VAD, time stretch or new downloads.

Physical noise sources (previously used non-official-test synthetic components):
- N1 `/data/datasets/dns_ins20/dns_snr_-20_-5/synthetic_snr-20_-5/noise/noise_fileid_5243.wav`
- N2 same directory `noise_fileid_6339.wav`.

Check their actual mono16k PCM16 headers, full byte hashes and prior frozen recipe source hashes where available. If mismatched stop; do not silently replace a noise. Each utterance uses the contiguous noise slice `[80000:80000+L)` at5seconds; no wrap/repetition or noise-window search. These are historical reused noise recordings, not newly held-out noise or a verified new training partition.

For each entire utterance s (L samples), each raw noise segment n and prescribed q in[-18,0]dB:
`g=sqrt(mean(s^2)/mean(n^2))*10^(-q/20)`; `x=s+g*n`. Powers are float64 over the COMPLETE same L samples, not active speech/local phoneme SNR. Both powers must be finite and positive.

For each utterance, compute all four conditions FIRST; let M be the maximum absolute sample among s and every g*n and x in this four-condition group. Use ONE group scale `a=min(1,0.9/M)` for its clean and ALL four noisy/noise components. This fixed downward-only scale is a synthesis clipping safeguard, not an inference clean-aware normalization. Record all g,a,M. Input/noise scale selection uses no enhancement/ASR output. Actual inference only uses the native noisy peak, never s, a clean mask, transcript or local SNR.

Serialize `a*s` as one scaled-clean PCM16 control and `a*x` as each noisy PCM16 WAV using the same explicit floor/saturate32768 rule as the previous canonical derived WAVs. No actual saturation is allowed on synthesized inputs. Save raw float32 scaled-clean/noisy arrays separately; physical noise raw arrays may be reconstructed from locked source bytes,g,a. Report unquantized component SNR and actual float32/PCM residual bounds. Expected max quantizer error <=1 PCM16 unit (plus explicit float32 rounding bound); mixture residual relative to unquantized clean/noise <=1LSB plus that bound. Outside scope: no alternative quantizer/gain/filter/seed search to pass. Peak must be<=0.90004, headers mono16kPCM16, lengths unchanged; all16 cases required.

Freeze full input registry after CPU preparation: order ID, noise[N1,N2], SNR[-18,0]; basenames `{id}_{noise}_snr_m18.wav`/`..._snr_p0.wav`; source file/array/PCM hashes, published text raw hashes, all coefficients, lengths and actual measured recipe guards. No model loaded in this first phase. Root reviews input/driver freeze before model launch.

## Native enhancement and random state

Matched existing checkpoint `/data/coding/sgmse/joint_train_logs/sb61i0gj_RRSD/epoch=118-step=80000.ckpt`, backend EMA, train_type mode2, OUVE PC/reverse_diffusion/corrector none/N30/snr0.5, Router and all optional calibration/anchor/regen modifications off. No checkpoint or core source edits.

Prefer running unchanged `joint_enhance_by_dir.main` once over the16 frozen files with observer-only wrappers capturing the actual matched y_pre and pre-clipping backend raw output. Use its unchanged native loader/eval/frontend noisy-peak normalization/_match_length/model.enhance path; wrappers must pass original arguments/returns unchanged. Never replace the frontend with an old quantized cache or independently preprocess the same input again.

Base seed2026 and native stable basename formula `(2026+int.from_bytes(blake2b(basename,digest_size=4),'little')) mod2^32`; record all actual seeds before inference. Basenames include SNR, so random states differ across SNR cases: this pilot is NOT a causal isolation of SNR from random trajectory. Within each case the compared frontend is exactly the one used by that backend. Freeze registry before outputs, no reruns/seeds. Use the already-proven native entrypoint, assert source/weight locks unchanged from the successful replay where shared. Log native input peak, raw frontend/backend shapes/dtypes and normalization/reference geometry; record N30/NFE if observable without changing computation. Do not require different inputs to share xT or trajectories.

Retain raw float32 frontend/backend waveforms AND native PCM16 scored audio. Main's existing output clipping stays native; explicitly log pre-clipping count/peak and any serialization clipping for EVERY arm, do not silently renormalize or exclude clipped outputs. Create frontend PCM using the same native SoundFile conversion, recording exact method; no per-clip loudness boost or trimming. All16 outputs retained even if proxy quality is poor. Technical shape/source failures stop, not adverse accuracy results.

## Frozen ASR/scoring

After enhancement, release model/GPU memory before loading the one already-cached local Whisper tiny.en path. Reuse the exact clean-calibration decode options, weight/source hashes, tokenizer/assets and orthographic word/CER scorer/tie rule (lock source rather than redefine). No network, prompts, language detection, beam/temperature fallback or second word recognizer download. No phoneme-CTC truth judge.

Decode four scaled-clean controls ONCE (new amplitude/PCM is deliberately distinct from prior unscaled controls; keep earlier cache without rerunning it). Decode each actual noisy, actual native frontend PCM and native OFF PCM once:4+16*3=52 transcribe API calls maximum. Count internal decode APIs separately; do not call either count neural-forward count. No repeated decoding after error/high WER. Changed scaled-clean controls do not exclude corresponding cases; they flag a measurement confound, and all16 still remain. A technical decoding exception stops with preserved partial results.

Publisher TXT is always reference. Save raw text/segments/tokens and full frozen word/character alignments. Report all52 items, per-case S/D/I/N and WER/CER, word-count-weighted corpus and sentence means separately, stratified by noise and SNR. Within each case report OFF−frontend proxy difference and canonical reference-position transitions: bothH, frontendH→OFFnotH, frontendnotH→OFFH, bothnotH. These are deterministic ASR alignment transitions, NOT verified speech deletions/recoveries or time-localized phoneme events. Report insertions separately; alignment ambiguity and language-model priors remain limitations. Forbid using clean-ASR as pseudo-reference.

Optional basic waveform checks (finite values, lengths, RMS/peak and exact serialization) are required; perceptual or semantic quality is not inferred from energy. No new PESQ/DNSMOS/model-based metric is necessary for this measurement pilot. No manually verified intelligibility claims or manufactured listening labels.

## Budget, phases, records and exit

CPU input preparation:0models/0network, <=120seconds and <=12MB preparation directory; single preparation marker, failure preserved. Prepared script/synthetic tests first by Engineer, actual CPU recipe execution by Worker after Root approval. Freeze the resulting registry/hashes before model launch.

Future model phase (NOT authorized by this plan alone): at most16 frontend calls+16 OFF N30 calls+52 transcribe API calls, one enhancement load and one ASR load, fixed CUDA0, no fallback, <=420seconds supervised runtime (reserve finalization), <=40,000,000 total experiment bytes including inputs/preparation/logs/validation, zero network/training/test access. Memory/disk preflight; no concurrency with existing experiments. One model-phase STARTED marker, no automatic retries, source/frozen pre/post locks and independent CPU artifact/scoring verification. Source/driver issues must be diagnosed and archived, not silently retried. Preserve previous checkpoints, data, dirty worktree, compatibility bridge/listening service; no commit/push/cleanup.

Success is a complete reproducible measurement record with honest confounds, not a WER threshold. If no stable publisher-reference difference emerges, finish this pilot without changing its sources/SNRs/seed/ASR settings. If candidate differences emerge, human adjudication and separate genuinely unseen examples are required before any enhancement mechanism/loss proposal. The next decision is not automatically a larger training run, conditional-weight patch or an OOD four-grid.

Required preparation config.yaml/registry/proofs/reports and frozen execution files; eventual metrics.json/experiment_report.md with every condition retained, raw/PCM audio, raw ASR results, scoring/alignment tables, source locks, runtime/call/clip accounting and independent validation. Original raw metrics remain immutable after final reports are added.
