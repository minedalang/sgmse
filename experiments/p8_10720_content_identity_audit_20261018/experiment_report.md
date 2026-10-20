# Cached content identity audit

This compares recognizer outputs against the clean recognizer proxy. It cannot identify acoustic hallucinations or verify words.

| Condition | Nout | M | S | D | I | Edit proxy | M/Nout | <=100ms matches |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| clean | 14 | 14 | 0 | 0 | 0 | 0.0000 | 1.0000 | 14 |
| noisy | 0 | 0 | 0 | 14 | 0 | 1.0000 | undefined (empty) | 0 |
| y_pre | 18 | 2 | 12 | 0 | 4 | 1.1429 | 0.1111 | 2 |
| off | 31 | 6 | 8 | 0 | 17 | 1.7857 | 0.1935 | 3 |
| router | 31 | 7 | 7 | 0 | 17 | 1.7143 | 0.2258 | 3 |

Primary alignment is unconstrained Levenshtein; ties prefer match, substitution, deletion, insertion during forward suffix-DP reconstruction. The secondary diagnostic maximizes monotonic same-ID matches within 100 ms + 1e-6 ms; its threshold is fixed.

See metrics.json for consonant analysis, shift distributions, prior-sequence comparisons, unmatched output classifications and input SHA256 hashes; sequence_alignment.csv for every operation; timeline.csv for cached token runs and available confidence; fixed_time_clean_targets.csv for unchanged source rows at clean timestamps.

Times are approximate CTC runs using a 20 ms hop. Cached posteriors cover clean-target classes only. An unavailable output-class confidence is left blank. No output mismatch is labeled a hallucination. Research Lead interpretation follows separately.

The local feature extractor configuration has do_normalize=true. Its per-clip normalization can amplify very quiet input before recognition: decoded tokens do not establish audibility or intelligibility in the original waveform. Read-only source WAV RMS dBFS, peak and nonzero fractions are recorded in metrics.json.

## Research Lead interpretation and user observation

The user reports that Y_pre is almost inaudible while Router-off produces
many more speech sounds but remains unintelligible, and asks whether these
correspond to clean or are hallucinations. This is a nonblind single-listener
whole-clip observation, not an independent phonetic transcript.

The cached recognizer does not show a simple successful content restoration:
off has31 decoded tokens versus14 in clean, with6 exact-ID matches on one
minimum-edit path,8 substitutions and17 insertions; only3 monotonic same-ID
matches remain within the prechosen100ms tolerance. These are token-proxy
operations, not6 proven correct phonemes or17 proven hallucinations. They
are compatible with some preserved/shared speech structure plus substantial
unmatched structure, distortion or timing changes. Y_pre's18 decoded tokens
also cannot be interpreted as18 audible phones because per-clip normalization
is enabled and its original waveform is very quiet.

The primary listener-relevant comparison remains off versus clean at the
same coordinates; Router is secondary. Similar off/Router patterns point to
the shared backend rather than a demonstrated Router-specific cause.

## Late interval energy check and hallucination-risk candidate

Read-only `local_energy_check.py/json` additionally measures fixed intervals,
without new inference or a tuned speech detector. In clip1.3–2.0s, clean RMS
is−98.2605dBFS,87.13% of samples are exact zero and99.26% have magnitude at
most one PCM16 least-significant bit. Off RMS is−50.7875dBFS, a47.4730dB
difference. Off also has multiple recognizer tokens in this interval.

This is a concrete candidate for substantial output where the clean reference
has negligible waveform energy. It strengthens the question of extra
speech-like output and is more specific than a global posterior increase.
Energy alone does not identify speech, and cannot distinguish residual noise,
time smearing/delay, or generative speech artifacts. Clean's last nonzero
sample is near the clip endpoint, so do not infer a precise acoustic speech
endpoint from the final CTC token or exact nonzero-sample position.

## Optional human checks, not a blocking gate

`listening.html` provides original full-context playback and windows A0–0.55s,
B0.45–1.05s, C1.10–2.00s, with clean/off/Y_pre comparisons. C is particularly
useful for asking whether the backend continues with audible extra speech
after the clean content. The user may report same sound, different/extra
sound, or unclear, without phonetic transcription. Default playback preserves
original gain; optional RMS matching must be noted if used.

The optional question was sent in the session and queued in
`/data/coding/sgmse/LISTENING_TODO.md`. No local-event response has been
received yet. Other research continues; unsubmitted listening is not a
negative or positive result. Until identity is verified, use the term
unmatched or suspected extra speech-like structure, not a measured
hallucination rate. The specific5243 d positive case remains valid separately;
it does not imply every low-SNR backend output is content-correct.
