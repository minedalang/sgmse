# Fresh-CV synthetic-reverb smoke (2026-10-18)

## Question

Does a simple reverberation perturbation on speakers excluded from the prior manifests reproduce the Router degradation seen on the locked official with-reverb test set?

## Protocol

Four one-utterance speakers were selected from `cv_synthetic` after excluding speakers in the prior P7, speaker, and adapter manifests. The selected speakers were `reader_10714`, `reader_10717`, `reader_10777`, and `reader_10785`. Their original clean/noisy files were read only; the DNS dataset was not changed.

For each utterance, three temporary conditions were constructed in this experiment directory:

- `identity`: original clean/noise mixture;
- `short`: fixed causal FIR `[1, .55, .30, .16, .08, .04, .02]`;
- `medium`: fixed causal FIR `[1, .72, .55, .42, .31, .23, .17, .125, .09, .065, .047, .034, .024, .017, .012, .0085, .006]`.

Clean and noise were convolved separately, the noise was rescaled to preserve the original nominal SNR, and both signals were given a common peak scale of 0.95. The measured SNR error after writing the WAV files was below `2e-6 dB` per file.

The frozen 80k checkpoint was run with EMA, PC sampler, `N=30`, no corrector, `snr=0.5`, seed 2026. The two arms were Router-off (`rrsd_scale=0`) and Router (`rrsd_scale=1`). There were 24 trajectories total. No official final-test file was used and no DNSMOS run was performed.

## Results

Router minus off means, with four-utterance bootstrap intervals, were:

| condition | PESQ | STOI | SI-SDR (dB) |
|---|---:|---:|---:|
| identity | +0.00195 [-0.02923, +0.03306] | +0.00178 [+0.00066, +0.00289] | +0.5473 [+0.4826, +0.6255] |
| short | +0.01237 [-0.00160, +0.02368] | +0.00172 [+0.00080, +0.00264] | +0.5025 [+0.4467, +0.5755] |
| medium | +0.02108 [+0.00555, +0.03931] | +0.00184 [+0.00109, +0.00259] | +0.4759 [+0.4018, +0.5692] |

The Router interaction relative to identity was:

| condition | PESQ interaction | STOI interaction | SI-SDR interaction (dB) |
|---|---:|---:|---:|
| short | +0.01042 [-0.01256, +0.04426] | -0.000054 [-0.000392, +0.000284] | -0.04484 [-0.08376, -0.01446] |
| medium | +0.01913 [-0.01181, +0.05374] | +0.000067 [-0.000598, +0.000757] | -0.07145 [-0.12239, -0.04033] |

## Decision

This smoke **does not reproduce the official final-set pattern**. The synthetic FIR conditions make the Router's PESQ delta slightly more positive and its SI-SDR benefit smaller, whereas the official matched interaction was negative in both PESQ and SI-SDR. The sample is intentionally small and the FIRs are not measured room impulse responses, so this is not evidence that training-distribution mismatch is false in general. It is evidence that the simple fixed-FIR reverb hypothesis is not yet sufficient to justify a reverb adapter, SNR gate, or additional training.

Per the predeclared stop rule, do not expand this exact synthetic-FIR screen to 16/32 utterances. A future reverb experiment would need an independently sourced, fixed RIR protocol and a fresh CV manifest before any training is considered.

## Reproduction and artifacts

`make_reverb_cv_smoke.py` constructs the temporary audio, `analyze_reverb_smoke.py` computes paired summaries, and `metrics/` contains the six per-condition CSVs. `input_manifest.json` records source/output hashes and measured SNR. No original dataset file or checkpoint was modified.
