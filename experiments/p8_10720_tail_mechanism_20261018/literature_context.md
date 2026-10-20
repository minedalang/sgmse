# Tail artifact mechanism: literature boundary (checked 2026-10-18)

## Primary sources read

1. [ArtiFree, arXiv2509.19495v1](https://arxiv.org/html/2509.19495v1),
   sections2–3.2: discusses phonetic errors and hiss/breathing artifacts;
   uses multiple diffusion samples and embedding variance/centrality to
   detect risk or select output. Its clean-correlation selector is an
   analytical oracle. Its reported thresholds and benchmark results are not
   validated on this project or this utterance.
2. [PASE, AAAI2026 official paper](https://ojs.aaai.org/index.php/AAAI/article/download/40562/44523),
   sections3.1–3.3: denoising WavLM representation distillation with a frozen
   clean teacher and trainable student; final-layer phonetic and first-layer
   acoustic streams drive a continuous vocoder. This differs from attaching
   a frozen-feature loss to the existing Router.

## What these sources do and do not establish here

The existence of speech-like/hissing artifacts and seed-dependent output in
generative enhancement is established prior work, not a new project finding
or an original algorithm. A seed sweep, ensemble selector, WavLM loss or a
silence gate is not novel merely because the current example contains an
extra s-like sound. Correlation or embedding variability is not a ground-truth
phoneme label, and low seed variance does not exclude a systematic artifact.

The project-specific open question is whether a paired/noisy condition can
drive speech-like energy where the clean target and frontend are near floor,
while the frontend anchor and random trajectory are held fixed. The local
condition/anchor intervention pilot is designed to separate those paths.
It uses clean reference and changes the conditional input off the normal
frontend/noisy joint distribution: it provides a bounded mechanism test,
not a deployment method or proof of novelty.

## Competing explanations to retain

- A residual acoustic cue may be expanded by the conditioned score network.
- The learned prior may add a plausible fricative-like sound under uncertainty.
- Neighboring true speech may smear into the silent gap.
- Stochastic trajectory choice may alter how much the artifact manifests.
- Recognizer identity errors remain possible even when a human hears an
  extra sound. Noisy-blank and feature confidence do not prove irrecoverability.

## Next evidence gate

Replicate the condition-only test at the20dB noise-reduction case where the
user specifically identified s-like sound. If the same path dependence is
found, report a within-utterance, cross-SNR mechanism candidate. Require
independent utterances and speech-active false-suppression controls before
considering a learned correction. Do not optimize the frozen official test
set or suppress all weak/high-frequency frontend residuals: those may include
the very correct weak consonants the project aims to preserve.
