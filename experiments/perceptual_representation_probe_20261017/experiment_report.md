# P3a perceptual representation ranking probe

## Scope and data reuse

This is a read-only feature/metric analysis of the existing 32-utterance `rrsd_tcut_screen` development split. A 4-utterance smoke was completed before the full run. No inference, training, DNSMOS evaluation, data/checkpoint/model/production-path modification, or audio copy was performed. All existing WAVs were read in place.

Arms were noisy, Router-off, Router, `t_cut=0.05`, and selective. Existing PESQ/STOI/SI-SDR values were reused for all five arms. Per-utterance DNSMOS SIG/OVRL/BAK was reused only for Router, `t_cut=0.05`, and selective; no DNSMOS was imputed or recalculated for noisy/off. Router and `t_cut=0.05` DNSMOS came from `rrsd_tcut_screen`, while selective and its Router comparator came from the existing selective evaluation CSV. All rows were joined by filename.

## Feature definitions and rank statistic

Each arm was compared to its same-utterance clean reference using normalized waveform MSE, relative complex-STFT MSE, 80-band log-Mel L1 in dB, 40-band ERB L1 in dB, three-resolution log-magnitude STFT L1 and spectral convergence (FFT sizes 512/1024/2048), 100-Hz smoothed Hilbert-envelope relative L1, and envelope modulation-spectrum L1 in dB.

For each utterance, each feature distance ranked the candidate outputs against an existing quality metric. Reconstruction metrics used all five arms; DNSMOS metrics used only the three arms with per-file DNSMOS. The report gives mean utterance Spearman correlation between negative distance and quality, and within-utterance pairwise rank agreement (smaller feature distance predicts the higher metric). 95% CIs bootstrap utterances (20,000 resamples, seed 2026), preserving paired arms and avoiding between-utterance difficulty/speaker confounds.

## Main result

No representation ranked all quality dimensions well. Better values here mean higher agreement with that metric, not human preference.

| Feature distance | PESQ pair agreement (5 arms) | STOI pair agreement (5 arms) | SI-SDR pair agreement (5 arms) | DNSMOS SIG pair agreement (3 arms) | DNSMOS OVRL pair agreement (3 arms) | DNSMOS BAK pair agreement (3 arms) |
|---|---:|---:|---:|---:|---:|---:|
| Normalized waveform MSE | 0.781 [0.703, 0.853] | 0.878 [0.806, 0.944] | 1.000 [1.000, 1.000] | 0.208 [0.115, 0.312] | 0.208 [0.125, 0.302] | 0.438 [0.323, 0.562] |
| Relative complex-STFT MSE | 0.781 [0.706, 0.853] | 0.878 [0.806, 0.944] | 1.000 [1.000, 1.000] | 0.208 [0.115, 0.312] | 0.208 [0.125, 0.302] | 0.438 [0.323, 0.552] |
| Log-Mel L1 | 0.569 [0.491, 0.650] | 0.522 [0.456, 0.594] | 0.400 [0.400, 0.400] | 0.792 [0.688, 0.885] | 0.792 [0.698, 0.875] | 0.562 [0.438, 0.677] |
| ERB L1 | 0.569 [0.491, 0.647] | 0.522 [0.456, 0.594] | 0.400 [0.400, 0.400] | 0.792 [0.688, 0.885] | 0.792 [0.698, 0.875] | 0.562 [0.438, 0.688] |
| MR-STFT log-magnitude L1 | 0.588 [0.509, 0.666] | 0.553 [0.481, 0.631] | 0.438 [0.403, 0.488] | 0.802 [0.708, 0.885] | 0.802 [0.719, 0.875] | 0.573 [0.448, 0.688] |
| MR-STFT spectral convergence | 0.753 [0.666, 0.834] | 0.856 [0.784, 0.922] | 0.878 [0.806, 0.944] | 0.375 [0.250, 0.510] | 0.333 [0.219, 0.458] | 0.417 [0.292, 0.542] |
| Envelope relative L1 | 0.619 [0.534, 0.703] | 0.647 [0.566, 0.731] | 0.531 [0.459, 0.609] | 0.688 [0.552, 0.812] | 0.688 [0.562, 0.802] | 0.521 [0.406, 0.635] |
| Modulation-spectrum L1 | 0.756 [0.675, 0.831] | 0.841 [0.762, 0.912] | 0.863 [0.781, 0.934] | 0.385 [0.260, 0.521] | 0.385 [0.260, 0.510] | 0.490 [0.375, 0.604] |

For OVRL, MR-STFT L1 had the highest pairwise agreement (0.802, CI [0.719, 0.875]); log-Mel and ERB were close (0.792, with CIs overlapping). These distances were poor at ranking PESQ/STOI/SI-SDR. Waveform and complex-STFT distances ranked SI-SDR exactly in this candidate set but had only 0.208 OVRL agreement and negative mean utterance Spearman with OVRL (`-0.625`, 95% CI `[-0.797,-0.438]`). DNSMOS-BAK was weakly ranked by every feature (all pair-agreement CIs include 0.5). No single feature is a reliable joint perceptual/fidelity ranking.

## Alignment and sensitivity sanity

All 32×5 waveforms were 16 kHz mono and had the same 480,000-sample duration. The estimated best lag within ±25 ms was exactly zero for all 160 output/reference pairs. Output-to-clean RMS ratios averaged 0.918–0.938 for enhanced arms and 4.77 for noisy input.

On the four smoke references, a clean-only gain perturbation of ×0.8 or ×1.2 produced normalized waveform/complex-STFT MSE of 0.04 and changed log-Mel L1 by 1.58–1.94 dB. An 80-sample (5 ms) shift produced wave/complex distances around 2.09, MR-STFT L1 2.72 dB, and log-Mel L1 1.48 dB. Thus these distances are not gain- or shift-invariant; the comparison intentionally used the existing time alignment and level, with no per-output alignment/gain normalization that could hide actual enhancement errors.

## Limitations and exit decision

This is an exploratory, repeatedly used development split with only three candidate enhanced arms for DNSMOS. The quality targets are objective/model-based metrics; there was no blind listening or independent human preference label. DNSMOS ranking agreement is therefore not evidence of perceptual preference, and its use as both target metric and proxy check is not independent validation. Pairwise values among three arms are coarse (at most three comparisons per utterance). WavLM/CDPAM/NOMAD were not evaluated.

**Negative/exit conclusion:** the simple auditory features show a promising association with DNSMOS SIG/OVRL but do not preserve ranking for PESQ/STOI/SI-SDR, while waveform/complex errors show the reverse trade-off. BAK remains poorly ranked. The result does not establish that a perceptual representation is a dependable training target or that it tracks listeners. Do not train a custom encoder or perceptual loss from this probe. Treat P3a as inconclusive for a deployable perceptual objective; only a small blinded paired listening check or an independent split with preference labels should precede P3b/P5a.

## Artifacts

`per_file.csv` stores all 160 feature/quality rows, including missing DNSMOS fields where no existing score was available. `metrics.json` contains descriptive means, utterance-bootstrap rank summaries and sensitivity checks. `smoke_per_file.csv` and `smoke_metrics.json` preserve the four-file alignment/sanity run. The standalone script is `perceptual_representation_probe.py`.
