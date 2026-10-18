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

## Noisy-cue follow-up

The same windows were compared directly between noisy and clean using raw-waveform correlation and normalized log-spectrum cosine. These are alignment-sensitive acoustic proxies, not phoneme labels.

| Window group | n | noisy-clean waveform corr (median) | noisy-clean spectrum cosine (median) | Router−off RMS (median) |
|---|---:|---:|---:|---:|
| All clean-active windows | 181 | 0.187 | 0.944 | −0.401 dB |
| `Y_pre` suppression windows | 55 | 0.098 | 0.912 | −1.999 dB |
| Other windows | 126 | 0.356 | 0.964 | −0.205 dB |

The candidate suppression windows have substantially weaker noisy-clean alignment than the other clean-active windows. This is evidence that the difficult intervals often contain less directly usable acoustic evidence, so the current data do **not** support the stronger claim that the frontend is simply deleting recoverable phonemes. The result is compatible with an information-limited regime, while leaving a smaller frontend-over-suppression possibility open.

Router-off is the arm that supplies most of the energy fill. In the candidate windows, Router is a median 1.999 dB below Router-off, while its spectral-cosine change relative to Router-off is essentially zero. Thus the Future Router increment is not the main source of the observed smoothing and may slightly reduce it on these examples. This is a local diagnostic result, not a global quality claim.

The follow-up output is `noisy_cue_windows.csv` and `noisy_cue_summary.json`.

## Clean-aware local SNR diagnosis

For diagnosis only, the aligned synthetic noise (`noisy - clean`) was used to estimate local SNR. This clean-aware quantity was not used for inference or threshold selection.

| Window group | n | Local SNR median | Clean RMS median |
|---|---:|---:|---:|
| `Y_pre` suppression windows | 55 | −20.41 dB | −48.19 dBFS |
| Other clean-active windows | 126 | −8.30 dB | −49.87 dBFS |

The local SNR difference is about 12.1 dB, while clean RMS is nearly unchanged. Across all windows, local SNR correlates with `Y_pre` RMS relative to clean at `r = 0.492`; clean RMS correlates at only `r = −0.022`. This makes local input difficulty a substantially more plausible explanation for the suppression pattern than simple speech loudness.

This does not prove that no phonetic cue is recoverable in any individual interval, and it is not a deployable confidence feature because it uses clean. It does, however, lower the priority of a frontend-preservation intervention until a manually identified counterexample with strong noisy evidence is found.

The clean-aware outputs are `oracle_snr_windows.csv` and `oracle_snr_summary.json`.
