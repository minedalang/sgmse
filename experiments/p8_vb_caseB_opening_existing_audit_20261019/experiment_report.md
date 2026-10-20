# Existing B opening CPU audit — completed

One execution of the preregistered existing-audio audit; no retries or failures. Case B is p226_028_N2_snr_m18.wav, 67653 samples at 16000 Hz (4.2283125 s). Seven frozen intervals and every one of 212 consecutive 20 ms frames are retained; the final frame contains 133 samples. No smoothing, alignment, parameter search, sample selection, model import, ASR, training, network operation, new output audio, or intervention was performed. Optional plots were omitted; CSV values remain unfloored.

Audit measured 0.341485 s and independent CPU validation 0.854961 s under the shared timeout60s command. Final directory bytes are recorded in final_receipt.json (cap 2,000,000 bytes). Independent validation passed all 1766 CSV rows using a full complex FFT, independent dot-product RMS, and Python math.fsum noise projections.

## Source and serialization checks

25 targeted source/preparation locks match before, after, and independent validation. The inherited attempt02 metrics and its independent validation retain their original 260-source provenance; this audit does not rescan all 260 files or reload/hash full model weights. Locks cover feedback, sources.json, B four played WAV copies and their original sources, both raw outputs, input raws, original clean/noise, input registry, model execution/config/metrics metadata, and this audit implementation/plan. Mutable listening HTML/UI validation files are excluded. Only this new audit directory was written.

The saved exact user feedback SHA256 is 308eaca771c4a4b258a428cdb27c3f570580a42a89a5c1ec919724e52e700edc. All four played WAV copies equal their existing source bytes/hashes and are mono PCM16, 67653 samples/16kHz. Both frontend/OFF float32 arrays reencode in-memory with actual SoundFile0.14.0/libsndfile1.2.2 to exactly the played WAV bytes. Input float32 and floor-quantized PCM exactly reconstruct from original sources. Native length mapping is identity, NFE30, seed2418667005, noisy-reference normalization0.899993896484375; these are verified existing metadata, no new inference.

Physical N2 is noise_fileid_6339.wav[80000:147653), gain 12.379505719013439, group scale 0.12425738974272588. Raw noisy minus raw clean minus physical noise maximum residual is 2.9753792985e-08; played noisy minus unquantized clean/noise residual maximum is 3.05171043562e-05. These are float32 rounding versus PCM quantization, not unexplained audio transformations. Exact raw/PCM peak, extrema, zero counts/fractions, RMS and every frequency band are available in fixed_interval.csv and all_frames20ms.csv.

## All seven fixed intervals

RMS is unnormalized linear full-scale amplitude from actual played PCM. A fixed interval is a descriptive grid, not a speech/no-speech annotation.

| interval (s) | clean RMS | noisy RMS | frontend RMS | OFF RMS | clean PCM zero fraction | clean raw zero fraction |
|---|---:|---:|---:|---:|---:|---:|
| 0.0-0.1 | 0.00071424866 | 0.0245099096 | 0.000597336281 | 0.00385853972 | 0.0075 | 0.000625 |
| 0.1-0.2 | 0.000367224404 | 0.0366329694 | 0.000485154512 | 0.00288939897 | 0.030625 | 0.005625 |
| 0.2-0.3 | 0.000600649009 | 0.0403905274 | 8.85765805e-05 | 0.0020272335 | 0.0275 | 0.001875 |
| 0.3-0.4 | 0.00106337942 | 0.028568351 | 0.00045284104 | 0.00391720703 | 0.00625 | 0.00125 |
| 0.4-0.5 | 0.000381152212 | 0.0344710856 | 0.000367229951 | 0.00221533879 | 0.0475 | 0.005625 |
| 0-0.5 | 0.000675549692 | 0.0334021294 | 0.000433579102 | 0.00308541303 | 0.023875 | 0.003 |
| whole | 0.00637510675 | 0.051010545 | 0.00518698216 | 0.00580703048 | 0.011662454 | 0.00135988057 |

No clean fixed interval is sample-exact zero in either raw or PCM. In [0,0.5), clean raw contains 24/8000 exact zeros (0.003), while PCM contains 191/8000 (0.023875). Nonzero microphone/quantization samples do not establish speech; the user's perceived quiet opening remains compatible with these samples.

Power bands use rectangular real FFT and correct one/two-sided DC/Nyquist weights, normalized by N²; all band sums match time-domain mean square. DC, 1kHz/6kHz/7.8kHz sine-equivalent cosine and Nyquist calibrations passed. Full complex FFT independently verifies every fixed/frame band, including odd-length whole/partial intervals.

| interval (s) | frontend PCM [4k,8k] power | OFF PCM [4k,8k] power |
|---|---:|---:|
| 0.0-0.1 | 7.00654916e-10 | 2.18250548e-09 |
| 0.1-0.2 | 1.20125049e-10 | 6.17493482e-09 |
| 0.2-0.3 | 8.90670917e-10 | 4.79217848e-09 |
| 0.3-0.4 | 4.92286729e-10 | 1.1418525e-08 |
| 0.4-0.5 | 2.60996585e-10 | 1.56949668e-08 |
| 0-0.5 | 4.57548987e-10 | 5.34824069e-09 |
| whole | 9.59660442e-08 | 5.47537813e-07 |

The three original disjoint frequency bands are all retained separately in CSV/metrics. The displayed [4k,8k] sum is descriptive and does not detect phonemes.

Full-band zero-lag physical-noise correspondence below uses saved raw float32 waveforms (promoted to float64) without mean/lag/phase/filter fitting. Each cell is signed cosine / projected-power fraction (cosine²). Alpha and equivalent played-PCM projections are included for all seven intervals in fixed_interval.csv; zero denominators are explicitly undefined and synthetic zero tests passed.

| interval (s) | frontend raw cos / fraction | OFF raw cos / fraction | (OFF−frontend) raw cos / fraction |
|---|---:|---:|---:|
| 0.0-0.1 | 0.3850555 / 0.1482677 | 0.2183158 / 0.04766177 | 0.1753841 / 0.03075957 |
| 0.1-0.2 | 0.4223532 / 0.1783822 | 0.1899128 / 0.03606686 | 0.1315733 / 0.01731153 |
| 0.2-0.3 | 0.1339383 / 0.01793947 | 0.03347708 / 0.001120715 | 0.02817453 / 0.0007938044 |
| 0.3-0.4 | 0.3813957 / 0.1454627 | 0.2210587 / 0.04886695 | 0.1850445 / 0.03424145 |
| 0.4-0.5 | 0.4458969 / 0.1988241 | 0.1070321 / 0.01145586 | 0.03406558 / 0.001160464 |
| 0-0.5 | 0.3339849 / 0.1115459 | 0.148284 / 0.02198816 | 0.1082061 / 0.01170855 |
| whole | 0.02080299 / 0.0004327646 | 0.03290662 / 0.001082845 | 0.03579742 / 0.001281455 |

## Interpretation boundary

OFF has more RMS energy than frontend in every one of the five frozen opening bins and their union. The full [0,0.5) played-PCM RMS is 0.003085413 versus 0.0004335791; whole-clip RMS is 0.005807030 versus 0.005186982. This is an amplitude observation on the fixed grid, not a manual onset/offset, extra-syllable location, phoneme identity, intelligibility judgment, or a hallucination rate. No recognizer timestamps or forced alignment were used.

The opening raw OFF physical-noise projected-power fraction is 0.02198816, and OFF−frontend is 0.01170855. Low projection does not exclude phase-changing or nonlinear residual noise and does not prove a new voice or a noise-to-speech mechanism. High projection likewise would not establish causal content generation. No human exact time boundaries are available.

This is a single case selected after existing outputs and listening were inspected, not an independent confirmation set. Preserve both observations: A did not support the ASR-selected tail harm candidate and OFF syllable quality sounded much better; B had perceived frontend syllable loss and strange/unintelligible OFF syllables, including a perceived quiet opening in clean/frontend. Noise-to-speech remains the user's unproven mechanism hypothesis. No global high-frequency gate, noisy weighting, training, sample expansion, stopped16-case inference, or new research direction is authorized by these numbers.

## Reproduction and audit record

config.yaml is JSON syntax valid as YAML. audit.py refuses to rerun if STARTED.json exists; reproduce in a new explicitly authorized empty audit directory with the same plan/scripts/config/source locks. Run the recorded two-script command in execution_receipt.json with a 60s timeout, using the repository env Python, -B and single CPU BLAS threads. Read-only guards reject all process/network creation and writes outside this audit; model imports are explicitly rejected by the audit. BytesIO encoding produces no audio file.

No failure occurred in this audit. Preexisting native-pilot failure records were preserved and not retried. preflight.json records the dirty git state/resources and live compatibility bridge PID39452/port42084 and listening PID16022/port8766. SQLite read-only verification confirms this worker uses codex_tools_compat. Source locks pre/post, implementation hashes, exact output hashes, total bytes and the independent validation hash are retained for review.
