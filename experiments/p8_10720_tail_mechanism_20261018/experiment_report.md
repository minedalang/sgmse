# Archived 10720 acoustic tail probe

Completed CPU-only analysis of existing sources; no model inference.

The original 13.75–15.75 s clip ends within the full 30 s source. Extended context is 12.75–17.75 s; the fixed reported tail is 15.05–15.75 s.

Raw 20 ms Hann / 10 ms hop values are in frames.csv; fixed tail aggregates and 100 ms blocks are in metrics.json and blocks_100ms.csv. Clean before/after activity is retained without a speech detection threshold.

tail_clean_spectrum_scan.csv scans every tail frame against all clean frames in the five second context. References at or below three PCM16 steps RMS are excluded from valid winners; unfiltered winners and their floor flags remain available. Query frames near that floor have no valid winner.

Context lag searches normalize each overlapping window and report fullband and RMS envelope similarity over ±1 s. Positive lag means output later than clean. These correlations and spectral matches do not prove timing displacement or causal noise conversion. Periodicity is a voicing proxy, not phoneme truth.

Source hashes, decoded hashes, manifest checks, code hashes, environment, and git state are recorded in provenance.json. All 15 original source manifest entries and their listening clip slices were verified; all source WAV hashes were unchanged after analysis.

| Variant | Arm | Tail RMS mean | Highband fraction mean | Periodicity mean |
|---|---|---:|---:|---:|
| original | noisy | 0.026723842317782843 | 0.17239224444727183 | 0.2516290180475665 |
| original | clean | 1.1180497051154172e-05 | 0.3969135412309262 | 0.17553066589264146 |
| original | y_pre | 8.721381467475475e-05 | 0.13557438761502813 | 0.47911111176451815 |
| original | off | 0.0021682919395082305 | 0.2551924173346106 | 0.4387850720081062 |
| original | router | 0.0015172612087274776 | 0.2529971170032098 | 0.4413524911886471 |
| original | noise_source | 0.026723739628414185 | 0.17239234862320474 | 0.25162873725630225 |
| original | scaled_noise | 0.026723739628414185 | 0.17239234862320474 | 0.25162873725630225 |
| noise_minus10dB | clean | 1.1180497051154172e-05 | 0.3969135412309262 | 0.17553066589264146 |
| noise_minus10dB | noisy | 0.008450893585649108 | 0.17239312214520353 | 0.25162003041864267 |
| noise_minus10dB | y_pre | 4.322169025459101e-05 | 0.21285494036316505 | 0.3904455569627796 |
| noise_minus10dB | off | 0.0006860456124008939 | 0.393172756577068 | 0.31012702211907517 |
| noise_minus10dB | router | 0.0004946321447418007 | 0.38333321360370526 | 0.31071711379910577 |
| noise_minus10dB | noise_source | 0.026723739628414185 | 0.17239234862320474 | 0.25162873725630225 |
| noise_minus10dB | scaled_noise | 0.00845078848230906 | 0.17239234862320474 | 0.25162873725630225 |
| noise_minus20dB | clean | 1.1180497051154172e-05 | 0.3969135412309262 | 0.17553066589264146 |
| noise_minus20dB | noisy | 0.002672500844484895 | 0.17239626252619208 | 0.25159495056268205 |
| noise_minus20dB | y_pre | 2.697124697014029e-05 | 0.06542142671741957 | 0.45839133205817484 |
| noise_minus20dB | off | 0.00015132936632193412 | 0.3720638022661612 | 0.2886727875786302 |
| noise_minus20dB | router | 0.00011530633911860239 | 0.35333661720874143 | 0.28947899200362204 |
| noise_minus20dB | noise_source | 0.026723739628414185 | 0.17239234862320474 | 0.25162873725630225 |
| noise_minus20dB | scaled_noise | 0.002672373962841418 | 0.17239234862320474 | 0.25162873725630225 |

Scientific interpretation is reserved for the Research Lead; no human words or phoneme identities are inferred.

## Worker factual summary from saved outputs

No fixed windows or thresholds were changed. Values below aggregate exactly the saved frames; levels are the mean of frame dBFS values, not dB of mean RMS. Highband, flatness and periodicity are dimensionless acoustic proxies, not speech labels. The three PCM16 step floor is a conservative diagnostic flag, not an audibility threshold.

| Noise attenuation | Arm | Mean frame dBFS | 4–7.5 kHz fraction | Flatness | Periodicity | Above-floor frames / 70 |
|---|---|---:|---:|---:|---:|---:|
| original | y_pre | -82.82 | 0.136 | 0.053 | 0.479 | 25/70 |
| original | off | -56.26 | 0.255 | 0.052 | 0.439 | 70/70 |
| original | router | -59.52 | 0.253 | 0.048 | 0.441 | 70/70 |
| noise_minus10dB | y_pre | -88.88 | 0.213 | 0.104 | 0.390 | 5/70 |
| noise_minus10dB | off | -68.79 | 0.393 | 0.134 | 0.310 | 61/70 |
| noise_minus10dB | router | -71.81 | 0.383 | 0.129 | 0.311 | 48/70 |
| noise_minus20dB | y_pre | -92.53 | 0.065 | 0.092 | 0.458 | 0/70 |
| noise_minus20dB | off | -80.05 | 0.372 | 0.186 | 0.289 | 34/70 |
| noise_minus20dB | router | -82.26 | 0.353 | 0.185 | 0.289 | 17/70 |

Clean tail mean RMS is 0.0000112, mean frame level −99.83 dBFS, with 0/70 frames above the configured floor. Its normalized spectrum and periodicity should therefore not be interpreted as speech evidence. The archived noise tail highband fraction/flatness/periodicity are 0.172/0.196/0.252; scaling changes its level, not those normalized features.

| Noise attenuation | Arm | Noise RMS-envelope Pearson r | Noise mean band-power cosine |
|---|---|---:|---:|
| original | y_pre | 0.213 | 0.963 |
| original | off | -0.178 | 0.735 |
| original | router | -0.181 | 0.699 |
| noise_minus10dB | y_pre | -0.096 | 0.881 |
| noise_minus10dB | off | -0.070 | 0.692 |
| noise_minus10dB | router | -0.085 | 0.626 |
| noise_minus20dB | y_pre | -0.073 | 0.959 |
| noise_minus20dB | off | -0.247 | 0.130 |
| noise_minus20dB | router | -0.248 | 0.140 |

Noise similarities refer to the exact archived paired noise, scaled by the recorded alpha (positive scaling leaves these correlations and cosines invariant). These summaries do not prove noise conversion or exclude a nonlinear dependence on noise. Many Y_pre and 20 dB backend frames are near floor; their normalized comparisons need caution.

| Noise attenuation | Arm | Valid query frames | Median best cosine | Winner time range (s) | Winners outside original 13.75–15.75 s clip | Most frequent winner times (s: count) |
|---|---|---:|---:|---|---:|---|
| original | y_pre | 25/70 | 0.998 | 12.83–17.62 | 9 | 14.38: 2, 14.99: 2, 14.00: 2 |
| original | off | 70/70 | 0.997 | 12.80–17.73 | 62 | 12.89: 11, 17.33: 3, 12.87: 3 |
| original | router | 70/70 | 0.998 | 12.80–17.73 | 60 | 12.89: 11, 16.17: 4, 17.33: 3 |
| noise_minus10dB | y_pre | 5/70 | 0.849 | 12.87–14.39 | 3 | 12.87: 2, 14.38: 1, 14.39: 1 |
| noise_minus10dB | off | 61/70 | 0.936 | 12.80–17.74 | 58 | 12.89: 13, 12.87: 6, 17.46: 4 |
| noise_minus10dB | router | 48/70 | 0.970 | 12.80–17.64 | 45 | 12.89: 13, 12.87: 5, 17.14: 2 |
| noise_minus20dB | y_pre | 0/70 | none | none | 0 | none |
| noise_minus20dB | off | 34/70 | 0.843 | 12.87–17.65 | 34 | 12.89: 15, 17.49: 5, 13.20: 2 |
| noise_minus20dB | router | 17/70 | 0.847 | 12.87–17.65 | 17 | 12.89: 13, 16.55: 1, 17.65: 1 |

The clean context has 85 above-floor frames before 13.75 s and 155 after 15.75 s; after-clip above-floor frame centers span 16.17–17.74 s. Clean after-tail mean RMS is 0.002071 (maximum frame RMS 0.006371), substantially above the fixed tail. Thus the clip end cannot be treated as the end of all clean activity.

At 20 dB attenuation, all valid backend spectral winners are outside the two second clip, frequently at 12.89 s. The scan selects one of 499 clean frames using only 16 spectral bands, and winner locations are dispersed. Neighboring clean spectra therefore provide possible analogues but do not establish a shifted clean event. At 10/20 dB attenuation every arm’s full five second waveform and RMS-envelope best positive correlation occurs at zero lag; this supplies no evidence of an overall temporal shift. A local shift or smear is not excluded by that context-wide result.

Post-run verification additionally rehashed all listening clips, all loaded WAVs, source metrics/manifest, prepared config and probe script against recorded provenance: unchanged. No model/checkpoint is opened by the probe; runtime outputs are confined to this experiment directory. No experiment retry was performed.
