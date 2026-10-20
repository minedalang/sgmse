# P8 phoneme probe (2026-10-20)

## Status

Completed read-only CTC phoneme diagnostic for four aligned groups × five conditions (20 clips). No training, dataset modification, or checkpoint modification.

## Protocol

- Model: `facebook/wav2vec2-lv-60-espeak-cv-ft`; model weight was downloaded once to `/data/.rootcache/huggingface` with XET disabled and proxy `127.0.0.1:7890`; model SHA256: `3173bde9e9ce490fa0f989e413c42f25bc1820c020adc1e6b9b87025b3cfcc5e`.
- Dependency: `phonemizer==3.4.0`, system `/usr/bin/espeak-ng`; CUDA inference; seed 0.
- Decoder: greedy CTC argmax. `decoded_phonemes` is the collapsed IPA token sequence. Token posterior is the mean softmax probability across each emitted CTC run.
- Consonant proxy: mean frame posterior mass over a fixed IPA consonant-token set, plus emitted consonant-run count. These are model proxies, not phoneme ground truth or human intelligibility labels.

## Aggregate proxy summary

| condition | clips | mean argmax posterior | mean consonant posterior mass | mean emitted consonant runs |
|---|---:|---:|---:|---:|
| clean | 4 | 0.9513 | 0.2061 | 17.75 |
| noisy | 4 | 0.9064 | 0.0579 | 0.75 |
| y_pre | 4 | 0.8815 | 0.1480 | 11.00 |
| off | 4 | 0.9038 | 0.1968 | 16.25 |
| router | 4 | 0.8983 | 0.1921 | 15.25 |

The proxy shows condition-dependent decoding, but the four groups are too small for a confirmatory content claim. Empty or very short noisy decodes occur in several clips. Consonant posterior mass is an acoustic-model confidence proxy and must not be described as recovered consonants or WER.

## Interpretation against the listening observation

Relative to `Y_pre`, the shared diffusion path raises the mean consonant-posterior proxy by about `+0.0489` for Router-off and `+0.0442` for Router, and increases the emitted consonant-run count. This is consistent with smoothing or re-naturalizing residual speech. Router does not improve this proxy over Router-off: its mean consonant posterior is lower by about `0.0047` and its mean run count is lower by `1.0`.

The decoded IPA strings for Router-off and Router remain substantially different from the clean strings in most groups. Since no verified sentence transcript or human phoneme labels are available, these differences cannot be classified as hallucination, deletion, or true recovery. The safe conclusion is limited to: the back end produces more consonant-like acoustic-model evidence than `Y_pre`, while this probe provides no evidence that Router adds correct missing phonemes beyond Router-off.

An additional frame-alignment diagnostic is in `alignment_results.json`. On frames where the clean decode is non-blank, same-time argmax agreement averaged `14.5%` for `Y_pre`, `10.0%` for Router-off, and `12.0%` for Router. This is only a weak timing-sensitive proxy: diffusion outputs can shift phone boundaries and CTC frame alignment is not a transcript comparison. It reinforces that the long decoded strings should not be treated as clean-aligned phoneme recovery, but it does not establish that Router-off deletes content.

The exploratory clean-target posterior probe (`target_posterior_results.json`) gives mean probability assigned to the clean CTC argmax token on clean-active frames of `0.0206` noisy, `0.1150` `Y_pre`, `0.0989` Router-off, and `0.1055` Router. Three of four noisy clips have nearly zero target support; file-id 11999 is the only modest exception. Because this uses same-frame CTC pseudo-targets and only four groups, it is evidence against a broad, easily recoverable noisy phoneme reserve in this sample, not proof that no local evidence exists. It does not justify a noisy-centered model or a content-loss training run.

## Independent scientific review of the 11999 exception

File-id 11999 is **not an E1 counterexample** on this probe. Its noisy clean-target posterior is `0.0711`, only `+0.0040` above `Y_pre` (`0.0671`), while target top-1 fractions are identical (`2/32 = 0.0625`). Router-off and Router rise slightly (`0.0967/0.0972`, `3/32`), but this same-frame CTC change is not evidence of correct phoneme recovery. The decoded strings are not verifiably aligned to the clean sequence, and the nonblank posterior is itself affected by blank and boundary changes. Therefore 11999 remains a **weak candidate for one direct validation**, not a positive result.

The only justified next check for this candidate is a blinded, time-aligned human forced-choice test on 3–5 preselected consonant windows, with the clean phoneme, strong confusions, and randomized presentation across `noisy / Y_pre / Router-off / Router`. The E1 chain would require noisy performance above chance, a measurable `Y_pre` drop, and recovery by off/Router on the same windows. If that chain does not hold in at least 3/5 windows, treat 11999 as a proxy anomaly and stop the noisy-centered/content-recovery branch. This report does not authorize training or large-scale inference.

## Decoded phonemes

| group | condition | decoded IPA sequence | mean posterior | consonant mass |
|---|---|---|---:|---:|
| 00 | clean | `s ɛ l f w ʌ z k ə n eɪ d i ə n b aɪ b ɜː θ` | 0.9591 | 0.1701 |
| 00 | noisy | `t ɛ` | 0.9647 | 0.0331 |
| 00 | off | `ɛ l f ɛ s t ɪ m eɪ d ɪ ŋ d m aɪ w ɜː θ` | 0.9555 | 0.1678 |
| 00 | router | `ɛ l f ɛ z s t ɪ m eɪ d ɪ ŋ d m aɪ w ɜː θ` | 0.9483 | 0.1640 |
| 00 | y_pre | `ɛ l f w ʌ z k ə m j uː ɾ i b aɪ b ɜː θ` | 0.9261 | 0.1541 |
| 01 | clean | `k æ p t ɪ m aɪ ɹ iː k ɑː n s t ɪ t uː ʃ ə n ʌ v ð ɜː f ɹ ɛ n ʃ ɪ p` | 0.9281 | 0.2618 |
| 01 | noisy | `` | 0.8734 | 0.0653 |
| 01 | off | `ɛ t p e d a j iː b m a t i k r y k i ɲ o v ɑ b e tʃ e` | 0.8636 | 0.1946 |
| 01 | router | `k a t b e d a j iː b m a t i k r y k i ɲ o v ɑ b e tʃ e` | 0.8667 | 0.1918 |
| 01 | y_pre | `a p e v a v i d m a t i t i s u n u v o r p o n ʃ e` | 0.8560 | 0.1851 |
| 02 | clean | `iː v ə n ɪ n ð ə m oʊ s t p ɜː f ɪ k t k ə m j uː n ɪ ɾ i ð` | 0.9533 | 0.2504 |
| 02 | noisy | `i a m i n` | 0.9039 | 0.0753 |
| 02 | off | `ʃ l a m d ɛ ɾ d ə p ɹ eɪ t ɛ m iː n uː t` | 0.9091 | 0.1711 |
| 02 | router | `ʃ ʌ m d ɛ d ə p ɹ eɪ t ɪ m ɪ n ɪ` | 0.9095 | 0.1586 |
| 02 | y_pre | `ɐ m o z i i m i n k ə` | 0.9197 | 0.0783 |
| 03 | clean | `n d iː d s ɜː t ə n l i w ɪ l` | 0.9649 | 0.1421 |
| 03 | noisy | `` | 0.8836 | 0.0581 |
| 03 | off | `s ɪ ɾ i p ʌ n ɜː n t ə ð oʊ l d æ l v ɚ ɹ i d ə s ɪ s t ɚ w s k` | 0.8869 | 0.2539 |
| 03 | router | `s ɪ ɾ i p ʌ n ɜː n t ə ð oʊ l d æ l v ɚ ɹ i n ə s ɪ s t ɚ w ɪ ð` | 0.8685 | 0.2542 |
| 03 | y_pre | `m eɪ k ə t æ n t ʌ s v ɛ l k ɪ ʌ s ʌ` | 0.8241 | 0.1743 |

Full token runs, per-run posterior, input hashes, and runtime are in `metrics.json`.

## Reproduction

```bash
/data/coding/sgmse/env/bin/python experiments/p8_phoneme_probe_20261020/run_decode.py
```
