# P8 local content probe (2026-10-20)

## Question

Test the user's listening interpretation with a low-cost, read-only acoustic diagnostic: when `Y_pre` is unusually weak during clean-active intervals, does the backend add energy, and does its local magnitude spectrum move toward clean? This is an acoustic probe only; it cannot identify phonemes or prove content recovery.

## Protocol

- Source: the 8-utterance manifest from `p8_frontend_backend_audit_20261020`.
- Noisy input, clean reference, `Y_pre`, Router-off, and Router use the same centered 12 s interval (9--21 s).
- 32 ms Hann frames, 10 ms hop, aggregated into 0.5 s windows.
- Clean-active frames are selected acoustically from the clean RMS (30th percentile, with a 2% peak floor).
- `rms_db_vs_clean` is the median frame RMS ratio in a window.
- `spec_cos_clean` is cosine similarity of normalized log magnitude spectra to clean.
- A possible frontend suppression window is `Y_pre RMS <= -8 dB` relative to clean.
- A possible backend energy-fill window additionally requires Router-off gain over `Y_pre >= 2 dB`.

## Results

There were 181 clean-active windows across the 8 utterances:

| Quantity | Result |
|---|---:|
| Possible `Y_pre` suppression windows | 55 / 181 (30.4%) |
| Suppression windows with Router-off energy fill | 48 / 181 (26.5%) |
| All-window median `Y_pre` RMS vs clean | −2.681 dB |
| All-window median Router-off RMS vs clean | −0.962 dB |
| All-window median Router RMS vs clean | −1.519 dB |
| Suppression-window median Router-off gain over `Y_pre` | +10.688 dB |
| Suppression-window median Router gain over `Y_pre` | +8.237 dB |
| Suppression-window median Router-off spectral-cosine change | +0.0011 |
| Suppression-window median Router spectral-cosine change | +0.0014 |

The strongest suppression case was fileid `10720`: 18 of 20 windows crossed the threshold, and Router-off gained a median 13.94 dB over `Y_pre`. Its median spectral-cosine change was −0.0023, so the large energy increase did not correspond to a consistent movement toward clean spectral structure.

## Interpretation

The probe is consistent with the user's description: in many weak-`Y_pre` intervals, the backend restores broadband/overall energy much more than it restores clean-aligned spectral structure. This supports a **smooth/round/residual-masking** interpretation over a content-recovery claim.

It does not establish that the backend deletes phonemes. The noisy RMS is dominated by the very low-SNR noise and is not evidence that usable phonetic cues remain. The local spectral cosine is also only a coarse acoustic proxy. The result therefore strengthens H8-B/H8-C, but cannot distinguish “frontend over-suppression” from “no recoverable information in noisy input.”

## Limitations and stop decision

- 8 utterances, one listener, and clean-aligned acoustic windows; no human transcription ground truth.
- The thresholds are diagnostic thresholds, not tuned model-selection criteria.
- Energy filling can be useful perceptually while still omitting a syllable.
- No training, new inference, dataset modification, or checkpoint modification was performed.

Do not start WavLM/Harm, GAN, or content-loss training from this probe. The next minimum-cost test is to inspect the same candidate windows in the noisy spectrum and then separate `Y_pre → Router-off` from `Router-off → Router` locally. Only if noisy retains clean-aligned cues should a content-preserving intervention be considered.

## Reproduction

```bash
/data/coding/sgmse/env/bin/python3.10 \
  experiments/p8_local_content_probe_20261020/probe_local_content.py
```

Outputs: `window_metrics.csv` and `summary.json`.
