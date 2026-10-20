Clean-text measurement calibration: completed

All four frozen official-derived clean utterances were transcribed once. Under the preregistered normalization, the single cached Whisper tiny.en recognizer matched all 39 publisher-reference words and 177 scoring characters: corpus micro WER/CER=0 and sentence mean WER/CER=0. This is technical completion and a limited clean measurement result, with no accuracy threshold. No listening judgment or enhancement result was collected.

| ID | Word S/D/I/N | WER | Character S/D/I/N | CER |
|---|---|---:|---|---:|
| p226_027 | 0/0/0/12 | 0.0% | 0/0/0/51 | 0.0% |
| p226_028 | 0/0/0/9 | 0.0% | 0/0/0/51 | 0.0% |
| p287_037 | 0/0/0/9 | 0.0% | 0/0/0/36 | 0.0% |
| p287_054 | 0/0/0/9 | 0.0% | 0/0/0/39 | 0.0% |

Original reference and hypothesis strings are retained verbatim in outputs/*_raw.json, outputs/*_scoring.json and metrics.json. The following display uses JSON escapes to preserve leading/trailing whitespace:

- p226_027 reference: `"They should have a major rethink about the event for next year.\n"`
- p226_027 hypothesis: `" They should have a major rethink about the event for next year."`

- p226_028 reference: `"Suggestions of the action being a punishment were dismissed.\n"`
- p226_028 hypothesis: `" Suggestions of the action being a punishment were dismissed."`

- p287_037 reference: `"But, after a dip in sales, growth was recorded.\n"`
- p287_037 hypothesis: `" But, after a dip in sales, growth was recorded."`

- p287_054 reference: `"In time, the First Minister will grow in stature.\n"`
- p287_054 hypothesis: `" In time, the first minister will grow in stature."`

References are original publisher TXT bytes from the verified training ZIP, not clean-ASR pseudo-labels. Both reference and hypothesis use the frozen lowercase/apostrophe/regex rules; word alignment is unit-cost Levenshtein with diagonal>deletion>insertion tie order. Character scoring concatenates [a-z0-9] characters. Leading spaces, terminal newline and the capitalization difference in “First Minister” remain visible in the originals and are removed by the preregistered scoring rule. Full word and character alignments are retained. Aggregate H=39/S=D=I=0 for words, H=177/S=D=I=0 for characters; micro and unweighted sentence means are separately preserved.

One execution began 2026-10-19T03:19:30.407910+00:00, session3199, supervisor PID44420 / child PID44423, and completed in 6.099228171 seconds, exit0. One model instance, four full-utterance transcribe API calls, four internal decode API calls, four audio decodes, no retries or failures. API counts are not autoregressive neural forward counts; those were not measured. No new model was downloaded. Exact commands and original metric/validation hashes are in execution_provenance.json.

Model /data/.rootcache/whisper/tiny.en.pt SHA256 d3dd57d32accea0b295c96e26691aa14d8822fac7d9d27d5dc00b4ca2826dd03; Whisper20250625, Torch2.14.1+cu130, NVIDIA L40S CUDA:0, fp32, seed20261019. Decoding used language=en, task=transcribe, single temperature(0.0,), beam_size1, supplied best_of1, condition_on_previous_textFalse, initial_promptNone, no_speech_thresholdNone, verboseFalse, word_timestampsFalse, fp16False. Installed code removes best_of at zero temperature, yielding effective best_of=None. BeamSearchDecoder(1) is not mislabeled GreedyDecoder. Empty native prompt=[] contained no publisher reference. Installed defaults, effective pre-inference options and each actual internal decode option set are recorded and matched. Entire stored PCM16 inputs retained amplitude, length and order; no gain, trim, resampling or VAD. Native Whisper internal log-mel/padding was retained.

Prepared independent standard-library CPU validator ran once after completion, exit0, command wall0.211473794 seconds, status=passed. It independently reconstructed all original TXT bytes, normalization, canonical full word/character alignments and corpus aggregates; checked all four input WAV hashes, actual decode options, counts, 37 source locks, 9 prepared locks and 22 original run artifact hashes. It performed 0 network calls, 0 models and 0 audio decodes. Original metrics.json and frozen files remain byte-identical, with metrics SHA256 f443996cd8c5e80d0be224ab8fcbebcde925bec65cf9d87d045ebbb10ecfef6a. Source pre/post/supervisor-post locks agree. Original artifact_manifest and artifact_bytes_before_final are snapshots preceding final metrics/report/validation/provenance, not a final directory total.

Runtime network attempts=[], network bytes0; enhancement/frontend/backend/Router/CTC/training calls0, no noisy or test content access. Bounds were180seconds/5,000,000 total artifactbytes/0network. Full git status before/after was unchanged, dirty files preserved. Listening server PID16022 and compatibility bridge PID39452 remain running. Old waveform-gate metrics, source data, derived files and weights were not edited; the original waveform conversion gate remains waveform_gate_failed and was not rerun.

These two speakers/four already selected clean items establish only the observed recognizer/publisher-reference agreement under this fixed measurement. A single ASR may have biases or pretraining exposure; no human transcription adjudication was performed. Zero clean WER does not prove actual spoken truth, phoneme timing, independent holdout/generalization, reliability under DNS low SNR, or enhancement benefit. It does not repair the failed historical local-conversion gate or justify pairing the new official-derived reference with local VB noisy. No further decoding, reselection, setting search, pilot, new audio or listening package was created; subsequent scientific design belongs to Research Lead.

Final experiment artifact bytes including this report, validation and execution provenance: 364400, below5,000,000.
