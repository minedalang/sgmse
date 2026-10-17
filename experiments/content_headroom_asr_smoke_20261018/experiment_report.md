# Content headroom ASR smoke (2026-10-18)

## Status

Completed read-only 8-utterance pseudo-ASR diagnostic. No training, full evaluation, dataset modification, or checkpoint modification was performed.

## Protocol

- Source: P7 32-utterance development manifest; SHA256 `77b64db9da6a1ff372a22dcda4a6316743133f383103aaffb3dbce2c671c178e`.
- Selection: target SNRs `[-20, -18, -16, -14, -11, -10, -8, -6]`; for each target, nearest unused utterance, filename tie-break. Actual selected SNRs span -20 to -6 dB.
- Whisper `tiny.en` on CUDA, greedy English decode; model SHA256 `d3dd57d32accea0b295c96e26691aa14d8822fac7d9d27d5dc00b4ca2826dd03`.
- Runtime: 25.57 s for 40 transcriptions.
- Metric: word-level edit distance divided by clean transcript word count. This is a pseudo-WER/ASR stability measure, not ground-truth WER; clean audio transcription is only a reference.

## Aggregate result

| condition | mean normalized edit distance |
|---|---:|
| noisy | 0.5932 |
| router_off | 0.4323 |
| router | 0.4375 |
| cut005 | 0.4439 |

Router has mean pseudo-edit distance 0.4375 versus Router-off 0.4323 (difference +0.0052); cut005 is 0.4439. The paired Router-minus-off difference is `+0.00516`, with 5/8 utterances favoring Router and a paired utterance-bootstrap 95% interval `[-0.04344,+0.07571]` (200,000 resamples, seed 20261018). On this tiny exploratory sample, Router does not show a consistent content-preservation gain over Router-off.

## Per utterance

| file | SNR | noisy | Router-off | Router | cut005 |
|---|---:|---:|---:|---:|---:|
| 025_fileid_3545.wav | -20 | 0.803 | 0.887 | 0.901 | 0.930 |
| 000_fileid_2673.wav | -18 | 0.529 | 0.414 | 0.643 | 0.643 |
| 002_fileid_7247.wav | -16 | 0.988 | 0.481 | 0.444 | 0.457 |
| 022_fileid_9359.wav | -14 | 0.457 | 0.358 | 0.346 | 0.346 |
| 011_fileid_8207.wav | -11 | 0.372 | 0.308 | 0.231 | 0.256 |
| 001_fileid_2972.wav | -10 | 0.478 | 0.232 | 0.246 | 0.232 |
| 024_fileid_6209.wav | -8 | 0.520 | 0.147 | 0.088 | 0.088 |
| 003_fileid_4431.wav | -6 | 0.600 | 0.631 | 0.600 | 0.600 |

Full transcripts and input SHA256 values are in `metrics.json`. The eight items are a development smoke only; no claim about true WER or held-out content fidelity should be made.

## Reproduction

```bash
/data/coding/sgmse/env/bin/python experiments/content_headroom_asr_smoke_20261018/run_smoke.py
```
