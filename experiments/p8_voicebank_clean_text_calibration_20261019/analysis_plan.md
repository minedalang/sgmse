# P8 VoiceBank clean-text measurement calibration — pre-inference plan

## Question and role

Can the already-cached word ASR produce useful, explicitly fallible publisher-reference comparisons on four source-defined clean utterances? This is a measurement pilot, not speech enhancement, an intervention, an independent confirmation set or a proof of spoken labels. The previous local-conversion gate remains failed; this experiment does not retest it.

Selection is all four previously metadata-selected valid-source IDs in frozen order: p226_027, p226_028, p287_037, p287_054. No replacement or exclusion based on transcripts or any ASR output. These four are only two speakers and have already served source preparation. Pretraining exposure is unknown.

## Immutable inputs

- For each ID use the entire mono 16k PCM16 `outputs/<id>_derived16k.wav` from `p8_voicebank_source_waveform_gate_20261019`, at stored amplitude, without trim, padding in the stored input, resampling, gain, VAD cropping or source substitution. Whisper's fixed internal log-mel/padding is allowed and recorded separately.
- The source gate's official 48k member, derived float array, PCM serialization, validation and file hashes establish this new explicit derivation chain. Do not claim equality to the old local16k clean or pair it with local VB noisy.
- Reference is each original publisher TXT from the verified training ZIP/manifest, never a clean-ASR pseudo-reference. Check raw bytes and original text against their frozen hashes. Only these four labels need be read by the runtime.
- Local model file only: `/data/.rootcache/whisper/tiny.en.pt`. Loading must use its explicit file path, disallow network and lock weight/source hashes. No new model or vocabulary downloads.

## Fixed decoding and budgets

One model instance, CUDA:0, eval/inference mode, float32 decoding (`fp16=False`), no automatic CPU fallback. Seed 20261019, recorded Python/NumPy/Torch/CUDA seed setup. Exactly one `model.transcribe` API call per utterance, at most four total. This count is not a claim of four autoregressive neural forward passes; record internal decode count if available without changing decoding.

Options: language `en`, task `transcribe`, temperature `(0.0,)` (one attempt, no temperature fallback), beam_size 1, best_of 1, condition_on_previous_text False, initial_prompt None, no_speech_threshold None, verbose False, word_timestamps False. Record all remaining installed defaults and the effective `DecodingOptions`; if the installed API cannot support these frozen options, stop before inference rather than silently substitute. Do not place publisher references in a prompt, use a vocabulary boost, change beam/normalization based on results, or decode a clip twice. Preserve empty output and segment/token information.

Hard limits: 180 seconds supervised runtime, 5,000,000 total experiment artifact bytes, zero network, zero enhancement/frontend/backend/Router/CTC-phoneme/training calls. Four independent complete-utterance ASR transcriptions only. Single STARTED marker, no automatic retries or overwritten successful results. Source and frozen files verified before/after. Existing datasets, checkpoints, dirty worktree, listening service and prior experiments remain unchanged.

## Frozen scoring

Store original reference and hypothesis separately from scoring strings.

Word tokenization for BOTH reference and hypothesis:
1. Replace U+2019 and U+2018 with ASCII apostrophe.
2. Lowercase.
3. Extract regex `[a-z0-9]+(?:'[a-z0-9]+)*` tokens in order. No number expansion, spelling correction, synonym mapping or contraction expansion.

Character sequence for CER is the concatenation of only `[a-z0-9]` characters from the lowercased text after the same apostrophe replacement. Thus spaces/punctuation/apostrophes are removed for CER, while an internal apostrophe remains part of a WORD token. This is an explicit simple orthographic metric, not a claim about perceptual equivalence.

Compute unit-cost Levenshtein with rows=reference, columns=hypothesis. Save one deterministic optimal full alignment with indices, tokens and operations. Among equal-cost backtrace candidates prefer diagonal (match/substitution), then deletion of reference token, then insertion of hypothesis token. Verify S+D+I equals edit distance, H+S+D=N_ref, H+S+I=N_hyp. Per-utterance WER=(S+D+I)/N_ref, CER analogously; WER may exceed 1 and must not be clipped. Empty hypothesis gives all deletions; empty reference is an explicit undefined rate, not a divided-by-one fake value. All four present references must have nonzero word/character length; an unexpected empty reference is a preparation error.

Report every raw reference/hypothesis, normalized strings/tokens, S/D/I/H and N, full word alignment, per-utterance WER/CER, corpus micro WER/CER (sum errors / sum reference lengths), and unweighted sentence means separately. Expected reference word counts from the frozen rule are 12,9,9,9, total39. No confidence interval or significance/generalization claim from these two speakers/four items.

## Interpretation and stopping

- Technical success: complete retained record of the four frozen transcriptions and independently recomputable scores under caps; no ASR-accuracy threshold is selected after the fact.
- Scientific outcomes: exact clean matches indicate feasibility only for these clean clips; clean errors reveal instrument/reference disagreement. Neither validates recognition on low-SNR/enhanced speech, proves publisher text error nor supplies manually verified phoneme timing.
- Preserve every discrepancy and unknown. No sample reselection, extra decoding, forced alignment or high-confidence word mining in this experiment. A later human audit may adjudicate disputed words but is a separately recorded task; do not invent listening results.
- If decoding is technically unsupported or fails, preserve failure and stop. If clean discrepancies are substantial, do not treat that as motivation to train an enhancement loss or as a need to search ASR settings. Reassess the measurement with Research Lead.
- Any natural matched noisy/Y_pre/OFF pilot needs a NEW frozen noise/SNR/input and model-call protocol; this clean calibration does not authorize it, nor a rerun of stopped VCV/5243 cases.

## Required artifacts

Prepared config.yaml, analysis_plan.md, frozen inputs/source hashes, deterministic scorer and tests, supervisor/driver, preparation metrics/report, frozen execution manifest. On execution: metrics.json, experiment_report.md, per-item raw decoding JSON, complete scoring/alignment records, execution provenance/log, source pre/post checks and an independent no-network CPU score verification. Preserve original raw metrics when adding independent verification.
