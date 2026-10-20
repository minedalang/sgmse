# Prepared only: clean-text measurement calibration

No Whisper/Torch/Numpy/SoundFile import, model load, audio decode, resampling, real network request or inference occurred in preparation. Standard-library WAV header reads and raw file hashes only. This is not a rerun or reversal of the failed source-conversion gate.

## Frozen design

Read the Lead-owned analysis_plan.md in full. Four IDs retain prior frozen order p226_027, p226_028, p287_037, p287_054; entire cached official-derived16k PCM16 WAVs at unchanged stored amplitude. Do not use unknown historical localclean conversion, noisy/test content or ASR-generated labels. Publisher raw text is extracted from the already-verified TXT ZIP for these four IDs only, checked against existing manifest SHA256/original text/byte counts, and kept separate from scoring text. Reference words are12/9/9/9,total39.

The explicit local model path /data/.rootcache/whisper/tiny.en.pt is passed to whisper.load_model, never the tiny.en alias. One CUDA:0 float32 eval model; no device fallback or download. Python/NumPy/TorchCPU/allCUDA RNG seeded20261019. A socket/URL guard is installed before heavy imports; AF_INET/AF_INET6 creation, connections, DNS and urllib URLs are refused, HF libraries set offline, and Whisper _download is replaced with refusal. NumPy waveform input avoids subprocess audio loading. No mean/RMS scaling, trim, resample, VAD or input padding; installed Whisper internal log-mel and fixed30s padding remain native.

Exactly one full-utterance model.transcribe call per item, maximum4. Temperature tuple(0.0,), beam_size1, best_of1, condition_on_previous_textFalse, initial_promptNone, no_speech_thresholdNone, languageen, tasktranscribe, verboseFalse, word_timestampsFalse, fp16False. Installed transcribe removes best_of when t=0, so effective DecodingOptions.best_of=None is recorded, not substituted with another beam. BeamSearchDecoder(1), not a claim of GreedyDecoder. Native empty prompt=[] is recorded separately from initial_promptNone; no publisher text is injected. Installed signature defaults and dataclass defaults/effective options are recorded before inference, and actual decode-call options checked. model.decode API calls are counted separately; no claim that four transcribes are four neural forwards.

## Execution/failure guards

run_calibration.py imports only standard library until the supervised child path. Frozen files/source hashes verified pre/post, including after child timeout. Single exclusive STARTED; refuses any prior start/output/results, no override or retry. Independent supervisor grants178seconds execution +2seconds finalization reserve within the180second budget and watches total artifacts with500KB final reserve under5,000,000bytes. Zero network. Exceptions preserve partial progress, traceback/log and source checks and stop; no same-command retry. All existing data, old gate files, dirty worktree and listening service are untouched.

37 immutable source locks include source-gate metrics/independent validation/config/selection/driver, all4official48kWAVs+4derived16kWAVs+4floatNPYs, publisher TXT ZIP/manifest, local model weight, all installed Whisper .py sources and tokenizer/mel assets. The new frozen manifest additionally locks the Lead analysis plan, scorer/driver, independent validator and preparation tests/records.

## Measurement and retained records

Separate raw reference/hypothesis, original ZIP bytes(hex/hash), complete raw Whisper result(text, segments, tokenids, avglogprob and other returned fields), input geometry/hash/amplitude statistics, actual options and internal decode count. Both word and character alignments preserve unit-cost deterministic diagonal>delete>insert backtrace, S/D/I/H/N, length identities, unclipped rates, explicit undefined empty-reference rates; required references are nonempty. Corpus micro versus sentence means are separate. No accuracy pass threshold, confidence interval, significance, selection, second decode or phonetic truth claim.

validate_completed_run.py independently recomputes normalization and canonical DP using standard library, checks raw publisher bytes, all per-item/full alignment and aggregates, source/preparation/artifact hashes without modifying original metrics. Reserved for Worker after successful execution; it cannot turn failed calibration into a successful one.

## Preparation tests

AST/safe import without heavy modules; installed DecodingOptions dataclass compiled from AST only (no Whisper import) and frozen API support checked;225 exhaustive independent sequence-pair alignments; apostrophes/case/regex/number/punctuation behavior; empty hyp/ref, WER>1, diagonal tie, micro versus sentence mean; exact raw publisher hashes/counts and37source locks. Synthetic wrong-hash, missing-supervisor and existing-STARTED refusal. Separate CPU process verifies socket creation/connect/DNS/URL all fail before network. All pass, zero audio/model calls. Test files are not scientific results.

metrics.json and experiment_report.md are absent and reserved for Worker. Root must review frozen files/hashes before a single execution. Clean ASR discrepancies measure this fallible recognizer versus publisher references only, not human truth, pretraining independence, low-SNR reliability or enhancement benefit. Keep all discrepancies; do not relabel from clean-ASR.

## Worker command after Root approval

`PATH=/data/coding/sgmse/env/bin:$PATH /data/coding/sgmse/env/bin/python /data/coding/sgmse/experiments/p8_voicebank_clean_text_calibration_20261019/run_calibration.py`
