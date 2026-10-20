# Existing case B opening audit — frozen CPU-only description

Triggered by the user's exact A/B listening feedback saved in `p8_voicebank_native_content_pilot_20261019/attempt02/listening/user_feedback_20261019.json` (SHA256308eaca771c4a4b258a428cdb27c3f570580a42a89a5c1ec919724e52e700edc). A did not support the ASR-selected tail content-harm candidate: syllables similar, neither extra speech at the end, OFF syllable quality much better. B was heard as frontend syllable loss and strange/unintelligible OFF syllables, including an opening perceived speech-free in clean/frontend. Noise-to-speech is a user's causal hypothesis, not an established mechanism.

Only B `p226_028_N2_snr_m18.wav`, full67653 samples/16kHz, already generated and scored. No model, ASR, training, downloads, new output audio, condition intervention, shifted alignment or seed/window search. Preserve previous four-sentence/16-case stop rule.

## Source proof

Use immutable parent `input_registry.json`, attempt02 `metrics.json`/independent validation, exact four played B WAV copies and their original sources, and saved raw frontend/OFF arrays. Scaled clean raw .f32le and scored clean PCM remain separate; reconstruct the actual N2 `[80000:147653)` physical-noise segment with original fixed gain/group scale from the registry. Check source byte hashes/header geometry, exact played WAV copies, all input quantization/mixture residual bounds and identical native length mapping/NFE30/noisy reference. Reencode saved frontend/OFF using actual libsndfile in memory for byte equality; do not assume ideal float64 floor matches native float32 serialization.

## Fixed intervals before looking at new numbers

Five consecutive100ms intervals: [0,.1),[.1,.2),[.2,.3),[.3,.4),[.4,.5), plus their union[0,.5) and whole[0,67653/16000). These are a descriptive fixed opening grid, NOT presumed speech-free ground-truth intervals. User supplied no manual boundary. Do not change intervals after inspecting energies.

Also process EVERY nonoverlapping20ms frame across the entire clip (320samples, final partial frame retained with actual sample count). No smoothing or highest-energy frame selection.

For played clean/noisy/frontend/OFF report actual unnormalized RMS, peak, exact-zero count/fraction and sample ranges. For raw counterparts report the same fixed opening metrics, distinguishing serialization. Report fixed rectangular-RFFT Parseval power bands [0,4000),[4000,7500),[7500,8000]Hz with correct real-FFT DC/Nyquist weights and no frequency tuning; band power is not a phoneme detector. Validate full-band sum to time-domain mean square, with synthetic sine/DC calibration. A plot may show all-frame RMS and highband energy for the FULL clip plus explicitly fixed opening; any display floor must not change numeric zero values or imply silence labels.

## Restricted noise correspondence

On the SAME prespecified intervals only, compute the full-band zero-lag scalar projection onto reconstructed physical noise: alpha=<y,n>/<n,n>, signed cosine=<y,n>/sqrt(<y,y><n,n>), projected-power fraction=cosine². No mean, lag, phase, band or filter fitting; zero denominators are undefined. Include frontend,OFF and OFF−frontend; do not call high/low projection proof of noise-to-speech or hallucination. Phase-changing/nonlinear residuals remain compatible with low projection.

## Interpretation limits and deliverables

Explicitly check whether clean is sample-exact zero in each fixed interval, but never equate nonzero mic/quantization samples with speech or low energy with no speech. Exact human speech onset/offset and phoneme identity remain unavailable. Do not claim a particular interval is the user's extra syllable based on the plot or cached ASR. No new forced alignment or recognizer timestamps.

Source locks before/after, config.yaml, reproducible audit script, metrics.json, experiment_report.md, fixed_interval.csv, all_frames20ms.csv, optional compact plot and independent CPU arithmetic/source checks. Hard60seconds execution/2,000,000 total bytes/0models-network-audio-outputs; no retries or research-direction changes by Worker. One case selected by listening after prior output inspection, not a new independent confirmation set. A's positive quality observation is not altered. All findings are descriptive source/energy/correspondence checks, not objective proof of phonetic content or causality.
