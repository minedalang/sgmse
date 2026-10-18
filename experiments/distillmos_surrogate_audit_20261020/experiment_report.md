# Distill-MOS surrogate trust audit

## Question

Can the official compact Distill-MOS model serve as a training-time perceptual surrogate for a future NCSN++ adapter without immediately optimizing the wrong metric?

## Protocol

This was a read-only audit on the existing 32-utterance development split already used by the perceptual representation probe. Existing WAVs were read in place; no audio was copied, regenerated, or written. The five existing arms were evaluated: noisy, Router-off, Router, `t_cut=0.05`, and selective. Distill-MOS v7 was loaded from the official `distillmos` package (`0.9.1`) with the bundled weight hash recorded in `metrics.json`. Pairwise ranking agreement was computed within utterance and bootstrapped over utterances (10,000 resamples). Existing PESQ/STOI/SI-SDR and DNSMOS values were reused.

## Results

| Target metric | Distill-MOS pairwise agreement | utterance bootstrap 95% CI | pair-delta Pearson r |
|---|---:|---:|---:|
| PESQ | 0.675 | [0.588, 0.759] | 0.413 |
| STOI | 0.634 | [0.569, 0.703] | 0.917 |
| SI-SDR | 0.581 | [0.519, 0.647] | 0.945 |
| DNSMOS-SIG | 0.656 | [0.521, 0.781] | 0.030 |
| DNSMOS-OVRL | 0.635 | [0.510, 0.760] | 0.111 |
| DNSMOS-BAK | 0.490 | [0.354, 0.625] | 0.072 |

The strong correlation with STOI/SI-SDR and near-zero correlation with DNSMOS dimensions indicate that Distill-MOS behaves more like a reconstruction/content proxy on this StoRM development set than a reliable predictor of the perception endpoint that currently conflicts with Router. This does not establish human listening alignment: both Distill-MOS and DNSMOS are learned metrics, and the split is not independent.

## Decision

Do not use Distill-MOS alone as the quality reward in a perceptual trust-region adapter. A loss that increases Distill-MOS could simply reproduce the already-favored reconstruction direction and fail to address the DNSMOS tradeoff. No adapter training was started. To reopen this route would require an independent human-preference or independently validated quality target and a multi-objective guard; the existing data do not justify that cost.

The result is a useful negative screening result for the proposed innovation direction, not evidence that Distill-MOS is generally invalid.
