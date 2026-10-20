# Masked same-speaker InfoNCE short validation

A new optional `--contrastive_mask_same_speaker_negatives` flag masks valid same-speaker off-diagonal entries from both paired InfoNCE denominators. The default remains disabled for backward compatibility. Unknown speaker IDs are never masked. Unit checks cover legacy equivalence, mask activation, gradients, and clean-target detachment.

A 1,000-step run initialized from the 80k EMA Router used `contrastive_weight=10`, `contrastive_speaker_weight=0`, UniSpeech-SAT hidden-mean pairing, batch size 8, LR 3e-5, seed 2026. It completed successfully with train loss 3080.870. Checkpoint: `joint_train_logs/5l5cb2a0-rrsd_speaker_pair_masked1000/epoch=0-step=1000.ckpt`.

An 8-utterance/4-speaker N=30 CUDA diagnostic gave the following means:

| arm | CAM++ cosine | PESQ | STOI | SI-SDR |
|---|---:|---:|---:|---:|
| Router | 0.909520 | 2.109723 | 0.850771 | 9.422118 |
| Unmasked pair | 0.910018 | 2.117692 | 0.850338 | 9.317884 |
| Masked pair | 0.910115 | 2.120045 | 0.850270 | 9.300570 |

Masked minus Router was +0.000596 CAM++ cosine, +0.010322 PESQ, -0.000501 STOI, and -0.121547 dB SI-SDR. This is too small a subset for a claim and does not remove the reconstruction trade-off. A full 32-file evaluation is deferred unless a matched-budget training comparison is first justified.

## Full 32-file follow-up (2026-10-17)

The checkpoint was subsequently evaluated on the full repeated-speaker CV manifest under CUDA, EMA, RRSD, N=30, SNR=0.5, seed 2026. Results are stored in `experiments/rrsd_cv_repeat32/masked_pair32_metrics.json`.

Masked versus the 80k Router anchor: CAM++ enhanced-clean cosine `+0.001613` (95% CI `[+0.000392,+0.002870]`), PESQ `-0.001500` (CI `[-0.006685,+0.003938]`), STOI `-0.000300` (CI `[-0.000553,-0.000071]`), and SI-SDR `-0.144008 dB` (CI `[-0.163897,-0.124513]`). The independent same-speaker cross-utterance CAM++ mean was `0.840694`, below Router `0.842322`. The masked run therefore does not establish an independent speaker-preservation gain and should not be expanded without a new matched-budget hypothesis.
