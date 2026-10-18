# Frontend state representation probe

Read-only CUDA probe using the frozen 80k EMA checkpoint and the first 8 rows of the fresh-CV manifest. The first 8 rows contain 4 speakers with two utterances each; this is not an 8-speaker sample. No training, checkpoint writes, dataset changes, or WAV generation occurred.

Three StormFrontend backbone leaf hooks were successfully registered (`act`, `output_layer`, `pyramid_upsample`) and executed for noisy and clean inputs. Hook shapes and pooled mean/std are stored in `per_utterance.json`. The frontend output was also computed through `_frontend_forward`.

Mean input SNR was −13.75 dB and mean noisy-clean waveform cosine was 0.2122. This is a representation extraction smoke test only. An exploratory OOF follow-up reused the existing rollout cache for the same eight filenames; its low-power results and limits are reported below. The first attempted script failed after selecting distinct speakers because `rec` initialization was placed inside the loop; this was corrected and rerun with the required first 8 rows.

Runtime was 10.46 s on CUDA. No official locked test set was accessed.

## Rollout-state OOF diagnostic

The existing 32 cached rollout states (8 utterances × 4 selected times) were reused; no PC trajectory or WAV was regenerated. Frozen EMA score forward produced a four-state mean clean-x0 NMSE target (mean 0.91538). A Ridge model using hook pooled statistics plus SNR, noisy RMS, and noisy-clean cosine was evaluated with leave-one-speaker-out OOF across the 4 speakers. Exploratory results: MAE 0.23319 versus constant-mean baseline MAE 0.33252, R² 0.1586, Spearman 0.6667. The sample is only 8 utterances and 4 speakers; this is low-power diagnostic evidence and does not establish cross-speaker explanatory power. No valid independent OOF label beyond the cache target was used.

## Exploratory rollout-label OOF follow-up

The existing 32 cached rollout states (four selected states for each of the same eight filenames) were reused. Filename equality was checked against the hook manifest; no reverse trajectory was rerun. For each utterance, the target is the mean clean-`x0` NMSE over its four cached states. A fixed-alpha (`alpha=1`) speaker-leave-one-out Ridge was compared with a constant mean and with simple observable features.

| feature set | MAE | R² | Spearman |
|---|---:|---:|---:|
| constant mean | 0.33252 | 0.0000 | 0.0000 |
| SNR + noisy/clean cosine + frontend RMS | 0.25186 | 0.3172 | 0.5000 |
| hook summaries only | 0.27217 | −0.1813 | 0.7143 |
| simple features + hook summaries | 0.23319 | 0.1586 | 0.6667 |

The combined summaries reduce MAE by `0.01867` relative to the simple observable baseline on this eight-utterance sample, but the sample has only four speakers and the fixed linear model is high variance. Hook summaries alone are worse than the simple baseline by MAE. This is a feasibility signal, not evidence that frontend state representations provide cross-speaker explanatory power or a deployable controller. There is no independent audio-quality target in this probe, and no adapter or other training was started.
