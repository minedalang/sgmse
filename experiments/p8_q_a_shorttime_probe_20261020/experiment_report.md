# P8-Q-A short-time acoustic probe (2026-10-20)

## Scope

This read-only diagnostic revisits 181 existing 0.5 s windows from eight
paired DNS-Ins20 development utterances. It compares 10 ms and 20 ms frames
inside 55 `Y_pre` suppression windows with 126 other clean-active windows.
The clean waveform is used only for offline analysis of the known synthetic
mixture. No source data, checkpoint, or model output was modified, and no
model inference or training was run.

The local SNR is `20 log10(RMS(clean) / RMS(noisy-clean))`. The analysis is
therefore clean-aware and cannot be used as an inference-time gate. No text,
phoneme boundary, or phoneme identity is available locally.

## Results

| frame | group | active-frame local SNR median | frames >= -10 dB | windows containing >= -10 dB | windows containing >= -5 dB | windows containing >= 0 dB |
|---:|---|---:|---:|---:|---:|---:|
| 10 ms | suppression | -20.38 dB | 188/2075 (9.1%) | 24/55 | 10/55 | 5/55 |
| 10 ms | other | -6.96 dB | 2642/4594 (57.5%) | 106/126 | 97/126 | 86/126 |
| 20 ms | suppression | -20.05 dB | 178/2044 (8.7%) | 19/55 | 7/55 | 1/55 |
| 20 ms | other | -6.83 dB | 2629/4493 (58.5%) | 108/126 | 92/126 | 79/126 |

Within suppression windows that contained a 10 ms frame at or above -10 dB,
the median fraction of those frames where `Y_pre` was at least 8 dB below
clean was 0.367. The corresponding median in the other group was 0.0. The
same pattern appears at 20 ms (0.333 versus 0.0). Thus a small number of
short-time energy glimpses can coexist with broad local suppression, but the
probe does not show that those glimpses carry a particular /k/, /d/, or any
other correct phoneme.

## Interpretation and stopping decision

The result supports a bounded mechanism observation: suppression windows are
low-SNR at both temporal resolutions, while a minority contains relatively
stronger short-time clean contributions. This is compatible with a
time-frequency glimpse hypothesis, but it does not establish phoneme
discriminability or model recoverability. Normalized log-spectrum cosine,
energy, or a glimpse count cannot distinguish the correct consonant from a
nearby consonant or a generated artifact.

Consequently this experiment does not justify a deployment gate, content
loss, P8-N, or P8-R. The next valid step requires an independently verified
phoneme event or transcript and then a local 10/20 ms analysis tied to that
event. Until that material exists, the current result remains an exploratory
clean-aware diagnostic.

## Reproduction

```bash
/data/coding/sgmse/env/bin/python \
  experiments/p8_q_a_shorttime_probe_20261020/shorttime_probe.py
```

Outputs are `metrics.json`, `shorttime_frames.csv`, and
`shorttime_window_summary.csv` in this directory.
