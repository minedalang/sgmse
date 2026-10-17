# Post-hoc SNR strata of the locked final evaluation (2026-10-18)

## Question

Does the Router-versus-off trade-off in the locked official evaluation concentrate in a particular encoded input SNR range? This is a descriptive analysis of the already completed final evaluation. It was not used to choose a new threshold, retrain, or alter the frozen test protocol.

## Data and protocol

- Source: `experiments/final_eval_full_20261018/` and its locked manifest (`SHA256 4c91782fb1459bd14f2b8581b1f4b537ca98bafd907ecc961226883213bcfa2b`).
- 150 `no_reverb` and 150 `with_reverb` utterances; all three arms were already generated before this analysis.
- The integer SNR was parsed from each manifest noisy filename. It ranged from 0 to 19 dB in each split.
- Compared Router and `t_cut=0.05` against the pre-existing `off` CSV, for PESQ, STOI, SI-SDR, and DNSMOS SIG/OVRL/P808/BAK.
- Bootstrap intervals use 10,000 utterance resamples, seed `20261018`. The strata are small and overlapping post-hoc summaries, not confirmatory subgroup tests.

## Results

Mean paired deltas (arm minus off) for the main metrics are:

| split / SNR bin | n | arm | PESQ | STOI | SI-SDR (dB) | DNSMOS OVRL |
|---|---:|---|---:|---:|---:|---:|
| no-reverb / all | 150 | Router | +0.0264 | +0.00052 | +0.319 | -0.0002 |
| no-reverb / <=6 | 59 | Router | +0.0270 | +0.00079 | +0.377 | -0.0033 |
| no-reverb / 7–12 | 48 | Router | +0.0271 | +0.00047 | +0.294 | +0.0027 |
| no-reverb / >=13 | 43 | Router | +0.0248 | +0.00021 | +0.267 | +0.0008 |
| with-reverb / all | 150 | Router | -0.0147 | -0.00215 | +0.164 | -0.0287 |
| with-reverb / <=6 | 59 | Router | -0.0039 | -0.00125 | +0.214 | -0.0425 |
| with-reverb / 7–12 | 48 | Router | -0.0115 | -0.00253 | +0.088 | -0.0272 |
| with-reverb / >=13 | 43 | Router | -0.0332 | -0.00295 | +0.180 | -0.0116 |

The `t_cut=0.05` arm has the same qualitative pattern. In with-reverb, its OVRL deltas are `-0.0332`, `-0.0245`, and `-0.0053` for the three SNR bins; its PESQ deltas are `-0.0017`, `-0.0264`, and `-0.0082`.

Utterance-bootstrap 95% intervals for the Router main deltas are recorded in `metrics.json`. For example, with-reverb OVRL is negative in all three bins, with intervals `[-0.0663,-0.0204]`, `[-0.0504,-0.0056]`, and `[-0.0342,+0.0117]`; with-reverb PESQ is most negative in the high-SNR bin, but its subgroup interval crosses zero because the subgroup is small.

## Interpretation and limits

The split effect is not explained by a single global SNR threshold. With reverb, OVRL damage is largest at low SNR, while PESQ damage is numerically largest at high SNR. This points to an interaction between reverberation and the type of error measured by each metric, rather than evidence for an SNR-only Router gate.

This result is post-hoc and uses the official final set, so it is not a basis for retuning `t_cut`, selecting an SNR threshold, or claiming generalization of subgroup effects. It also does not identify whether the cause is reverberation, front-end conditioning, phase/reconstruction behavior, or a DNSMOS artifact. No new inference, training, dataset modification, or checkpoint modification was performed.

## Reproduction

The read-only analysis script is `analyze_final_strata.py` with the SGMSE environment. The compact per-utterance deltas are in `per_utterance.csv`; complete statistics and bootstrap intervals are in `metrics.json`.
