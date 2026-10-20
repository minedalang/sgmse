# P3b literature and local objective audit

## Scope

This is a read-only audit of public method descriptions and the actual StoRM/SGMSE code. It was performed before downloading any perceptual model or starting a training run. No dataset, checkpoint, model parameter, or inference output was modified.

## Public method boundary

- [CDPAM (ICASSP 2021)](https://arxiv.org/abs/2102.05109) is a learned **full-reference** perceptual metric. Its central ingredients are contrastive/multi-dimensional embeddings and triplet human judgments, and the paper already reports using the metric in speech enhancement. Adding CDPAM as a loss would therefore be a comparison baseline, not a new contribution by itself.
- [NOMAD (ICASSP 2024)](https://research.google/pubs/nomad-unsupervised-learning-of-perceptual-embeddings-for-speech-enhancement-and-non-matching-reference-audio-quality-assessment/) learns an unsupervised embedding with NSIM-guided triplets and computes Euclidean distance against non-matching references. It is relevant as a pretrained non-matching perceptual baseline, but its embedding and training principle are already published.
- [Investigating Training Objectives for Generative Speech Enhancement (2025)](https://arxiv.org/abs/2409.10753) studies score-based and Schrödinger-bridge objectives and adds differentiable PESQ plus time-domain L1 to an SB data-prediction objective. On its VB-DMD table, its SB-VE M5 baseline reports PESQ `2.91` / SI-SDR `19.4 dB`; M6 with `alpha_P=1e-3` reports PESQ `3.70` / SI-SDR `8.3 dB`; lower PESQ weights M7/M8 recover some SI-SDR (`14.1` / `15.3 dB`) with PESQ `3.50` / `3.44`. These are the paper's own benchmark figures, not directly comparable with this project's DNS split. They show that perceptual gains can carry a substantial reconstruction tradeoff. Perceptual optimization in a generative speech-enhancement model is prior art; the objective itself is not a sufficient novelty claim here.

## Local code facts

The 80k Router checkpoint's embedded `hyper_parameters` were inspected directly on CPU with `torch.load(..., map_location="cpu")` and report:

```text
sde=ouve
loss_type=score_matching
loss_weighting=sigma^2
pesq_weight=0.0
frontend=storm
```

In `sgmse/model.py`, the differentiable PESQ branch is inside `loss_type == "data_prediction"`; the ordinary OUVE score-matching branch computes the sigma-weighted complex score loss. In `sgmse/joint_model.py`, enabling RRSD or frontend no-regret paths explicitly requires `score_matching` with `sigma^2`. Therefore PESQ is present in the codebase but is not active in the 80k Router objective. Enabling it requires a different data-prediction training objective and disabling the score-matching-only paths; it is not a drop-in loss ablation. Any comparison would need matched initialization, trainable parameter set, training budget, EMA, and sampler evaluation.

The local SGMSE environment has PyTorch, torchaudio, Transformers, SpeechBrain, librosa, and soundfile, but no importable `cdpam` or `nomad` package. No weights were downloaded because `/data` has only about 1.9 GB free and the P3b listening gate has not yet produced an independent distortion hypothesis.

## Decision

The literature audit rules out the following as stand-alone innovation claims:

1. adding a generic perceptual loss;
2. inserting CDPAM or NOMAD as a frozen distance;
3. adding a differentiable PESQ term;
4. adding an Adapter or changing a loss coefficient without a StoRM-specific mechanism.

The next defensible gate remains the blinded listening pilot in `experiments/p3b_blind_listening_pilot_20261017/`. If it identifies a repeatable StoRM-specific distortion, the smallest useful follow-up is a **backend-only**, matched-budget probe with the original score objective as the control and a published perceptual metric as a baseline. If it does not, the perceptual-training branch should be stopped rather than converted into a loss-weight sweep.

## Reproducibility and cost

The audit used local source inspection and public HTML pages only. It consumed no GPU time, did not launch inference/training, and downloaded no model weights into the repository.
