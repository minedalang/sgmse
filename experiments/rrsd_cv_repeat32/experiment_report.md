# Repeated-speaker CV diagnostic (16 speakers, 32 utterances)

This diagnostic selected two different utterances for each of 16 CV speakers and excluded `reader_10658`, the only train/CV speaker overlap found in the audit. The noisy inputs are symlinks under `/tmp/rrsd_cv_repeat32/noisy`; the original dataset was not modified.

All five arms used CUDA, EMA, N=30, SNR=0.5, seed 2026, and the same 32-file manifest. `off` and `router` use the 80k checkpoint. `pair`, `proto01`, and `proto01_cw1` use the existing 1,000-step speaker checkpoints, so their comparison is a diagnostic and is not a matched training-budget causal comparison.

## Mean metrics

| arm | CAM++ enhanced-clean cosine | PESQ | STOI | SI-SDR (dB) |
|---|---:|---:|---:|---:|
| off | 0.844619 | 1.678501 | 0.796901 | 6.482536 |
| router | 0.841249 | 1.691523 | 0.798124 | 6.966351 |
| pair | 0.842628 | 1.689088 | 0.797883 | 6.842032 |
| proto01 | 0.842649 | 1.688038 | 0.797876 | 6.840583 |
| proto01_cw1 | 0.841882 | 1.690266 | 0.798048 | 6.912326 |

Paired bootstrap uses 20,000 resamples with seed 2026. Deltas below are relative to Router:

| arm | cosine delta (95% CI) | PESQ delta (95% CI) | STOI delta (95% CI) | SI-SDR delta (95% CI) |
|---|---:|---:|---:|---:|
| off | +0.003370 `[+0.001306,+0.005366]` | -0.013022 `[-0.021294,-0.005442]` | -0.001224 `[-0.001820,-0.000689]` | -0.483816 `[-0.549329,-0.417427]` |
| pair | +0.001378 `[+0.000343,+0.002536]` | -0.002435 `[-0.007221,+0.002402]` | -0.000242 `[-0.000470,-0.000040]` | -0.124320 `[-0.141384,-0.107450]` |
| proto01 | +0.001400 `[+0.000331,+0.002552]` | -0.003485 `[-0.007965,+0.000307]` | -0.000249 `[-0.000482,-0.000046]` | -0.125768 `[-0.142994,-0.108634]` |
| proto01_cw1 | +0.000633 `[-0.000181,+0.001641]` | -0.001257 `[-0.005398,+0.002103]` | -0.000076 `[-0.000191,+0.000031]` | -0.054025 `[-0.062568,-0.045495]` |

## Cross-utterance speaker probe

The independent WeSpeaker CAM++ probe contains 16 same-speaker cross-utterance pairs and 480 different-speaker pairs. Enhanced-enhanced cosine means (same/different) were:

- off: `0.838524 / 0.245141`
- router: `0.842322 / 0.253854`
- pair: `0.840624 / 0.250419`
- proto01: `0.840928 / 0.250311`
- proto01_cw1: `0.841816 / 0.252081`

All arms preserve a large same/different separation, but none of the speaker-trained arms exceeds Router on same-speaker cosine. The speaker-trained arms also lose SI-SDR against Router, with the loss clearly below zero for all three. Because the speaker checkpoints were only 1,000-step runs while the Router is an 80k checkpoint, this is evidence against the current short-run speaker variants, not a final proof that every possible speaker training schedule fails.

Conclusion: this diagnostic does not support claiming an independent Speaker–Content improvement. Do not start a long speaker training run until a matched-budget checkpoint or a stronger speaker-specific hypothesis is designed.

## Masked same-speaker negative run (2026-10-17)

The 1,000-step checkpoint `joint_train_logs/5l5cb2a0-rrsd_speaker_pair_masked1000/epoch=0-step=1000.ckpt` was evaluated on the same 32-file manifest with CUDA, EMA, RRSD, N=30, SNR=0.5, and seed 2026. This arm only enabled `--contrastive_mask_same_speaker_negatives`; it did not change the dataset or the other inference settings.

| arm | CAM++ enhanced-clean cosine | PESQ | STOI | SI-SDR (dB) |
|---|---:|---:|---:|---:|
| masked | 0.842862 | 1.690023 | 0.797825 | 6.822344 |

Relative to Router, the paired bootstrap (20,000 resamples, seed 2026) was:

- CAM++ enhanced-clean cosine: `+0.001613`, 95% CI `[+0.000392,+0.002870]`;
- PESQ: `-0.001500`, 95% CI `[-0.006685,+0.003938]`;
- STOI: `-0.000300`, 95% CI `[-0.000553,-0.000071]`;
- SI-SDR: `-0.144008 dB`, 95% CI `[-0.163897,-0.124513]`.

The independent same-speaker cross-utterance CAM++ mean was `0.840694` for masked versus `0.842322` for Router; different-speaker means were `0.250515` and `0.253854`. Thus masking the false negatives did not produce an independent speaker-preservation gain and increased the reconstruction trade-off. Since this remains a 1,000-step checkpoint versus the 80k Router anchor, it rejects this short-run variant but does not reject every possible matched-budget speaker schedule.
