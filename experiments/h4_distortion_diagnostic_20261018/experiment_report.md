# H4 distortion proxy diagnostic (read-only)

## Scope
Compared existing aligned WAV outputs for `off`, `router`, and `cut005` on 32 development utterances. No inference, training, dataset modification, or checkpoint modification was performed.

## Signals
- Spectral flatness: mean Hann STFT flatness (512 FFT, 128 hop).
- Harmonicity: mean peak normalized autocorrelation over 80–400 Hz period lags.
- Modulation ratio: 2–20 Hz energy in log-RMS envelope spectrum.

These are output-level proxies, not perceptual ground truth.

## Inputs and confounds
- n = 32 matched files; input WAV SHA-256 hashes are in `metrics.json`.
- PESQ/STOI/SI-SDR are from existing `paired_metrics.csv`; DNSMOS SIG/OVRL are available only for router and cut005.
- This is a development/screen set; there are no clean/noisy WAVs in this directory, and content/SNR differences can confound correlations.

## Strongest associations
- `cut_minus_router` proxy `harmonicity` vs `cut_minus_router_si_sdr`: Spearman r=0.457 (p=0.00847), Pearson r=0.542, n=32.
- `router_minus_off` proxy `harmonicity` vs `router_minus_off_stoi`: Spearman r=0.431 (p=0.0139), Pearson r=0.616, n=32.
- `cut_minus_router` proxy `harmonicity` vs `cut_minus_router_stoi`: Spearman r=0.430 (p=0.014), Pearson r=0.391, n=32.
- `cut_minus_router` proxy `rms` vs `cut_minus_router_pesq`: Spearman r=0.386 (p=0.0291), Pearson r=0.141, n=32.
- `cut_minus_router` proxy `rms` vs `cut_minus_router_stoi`: Spearman r=-0.381 (p=0.0313), Pearson r=-0.482, n=32.
- `router_minus_off` proxy `rms` vs `router_minus_off_pesq`: Spearman r=0.365 (p=0.0397), Pearson r=0.175, n=32.
- `cut_minus_router` proxy `flatness_p90` vs `cut_minus_router_pesq`: Spearman r=-0.365 (p=0.0399), Pearson r=-0.311, n=32.
- `router_minus_off` proxy `rms` vs `router_minus_off_si_sdr`: Spearman r=-0.365 (p=0.0401), Pearson r=-0.349, n=32.
- `cut_minus_router` proxy `modulation_2_20_ratio` vs `cut_minus_router_pesq`: Spearman r=0.336 (p=0.0603), Pearson r=0.343, n=32.
- `cut_minus_router` proxy `flatness` vs `cut_minus_router_pesq`: Spearman r=-0.294 (p=0.102), Pearson r=-0.172, n=32.
- `router_minus_off` proxy `harmonicity` vs `router_minus_off_si_sdr`: Spearman r=0.291 (p=0.106), Pearson r=0.261, n=32.
- `cut_minus_router` proxy `flatness_p10` vs `cut_minus_router_DNSMOS_OVRL_e`: Spearman r=-0.286 (p=0.113), Pearson r=-0.265, n=32.

## Assessment
- `rms`: 0 qualifying association(s): []
- `flatness`: 0 qualifying association(s): []
- `flatness_p10`: 0 qualifying association(s): []
- `flatness_p90`: 0 qualifying association(s): []
- `harmonicity`: 3 qualifying association(s): [('router_minus_off', 'router_minus_off_stoi', 0.431), ('cut_minus_router', 'cut_minus_router_stoi', 0.43), ('cut_minus_router', 'cut_minus_router_si_sdr', 0.457)]
- `modulation_2_20_ratio`: 0 qualifying association(s): []

Harmonicity is the only weak exploratory lead: its per-utterance change is positively associated with Router/off STOI change (Spearman 0.431) and cut/Router SI-SDR change (0.457). This does not establish one artifact mechanism: the associations use different contrasts and different reference metrics, DNSMOS OVRL is not associated (Spearman -0.168, p=0.359), and the set is a 32-file development screen. No proxy met the predeclared requirement of a repeated, directionally coherent association across both contrast families including an independent DNSMOS signal. The observed correlations are exploratory and cannot support a GAN/discriminator causal route. **Recommendation: stop H4 GAN/discriminator development for now.** Revisit only with a fresh held-out set and a pre-registered artifact measure if a later mechanism requires it.
