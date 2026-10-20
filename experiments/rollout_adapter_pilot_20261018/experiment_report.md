# Rollout-aware backend adapter pilot (2026-10-18)

## Bottleneck hypothesis

The score network is trained on analytical forward OUVE states, while inference evaluates it on states produced by its own reverse sampler. The diagnostic question was whether the same frozen score model predicts clean `x0` less accurately on those rollout states.

## Diagnostic protocol

- Four P7 development utterances and an independent fresh-CV set of 8 utterances from 8 speakers were used; the official test set was not used.
- For each reverse step, capture the pre-predictor reverse state `x_t`.
- At the same time `t`, independently sample an analytical forward OUVE state from the clean/noisy conditioning.
- Run the same frozen EMA score model on both states and convert the score to a clean `x0` estimate using the configured OUVE weights.
- Compare normalized complex-spectral MSE to the clean target. This is a matched-time comparison; it does not assume reverse states should equal the forward marginal.

## Gap result

| set | rollout x0 NMSE | analytic-forward x0 NMSE | rollout minus forward | positive utterances |
|---|---:|---:|---:|---:|
| P7 smoke, 4 utterances | 0.8021 | 0.2147 | +0.5874 | 4/4 |
| fresh CV, 8 utterances | 0.5140 | 0.1356 | +0.3785 | 8/8 |

The fresh-CV gap is largest around `t≈0.73–0.77` and remains positive through the low-noise end. This identifies an observable, time-localized train–inference state mismatch. It is a bottleneck candidate, not proof of causality.

## Intervention

A zero-initialized residual backend adapter was inserted after NCSN++ module 33. It has 16,512 trainable parameters; the frontend, base score network, SDE, Router, and checkpoint were frozen. Cached rollout states used reverse steps `[3, 5, 7, 9]`. Six utterances were used for training and two were held out by utterance.

The first `lr=1e-3`/200-step pilot was unstable: holdout NMSE increased from `0.49464` to `1.05453` (+113.2%). A single-step gradient check showed loss `0.3441→2.9494`, so this setting was rejected and not repeated.

The bounded follow-up used `lr=1e-5`, 100 steps, and gradient clipping at 1.0:

- train NMSE: `0.77255→0.36800`
- holdout NMSE: `0.49464→0.28508` (`−42.4%`)

This establishes that the diagnosed mismatch is at least partially correctable by a small backend-only intervention on unseen utterances.

## Audio evaluation (training set plus 2-utterance holdout)

The exact zero adapter baseline and trained adapter used the same 8 utterances, seeds, PC sampler (`N=30`, `snr=0.5`), and no corrector. The adapter was fit on six of these utterances; only two were utterance-level holdouts. Thus aggregate 8-utterance metrics are partly in-sample and do not constitute an independent CV replication.

| metric | baseline | adapter | paired mean delta |
|---|---:|---:|---:|
| PESQ | 2.0079 | 2.0453 | +0.0374 |
| STOI | 0.8784 | 0.8800 | +0.0015 |
| SI-SDR (dB) | 9.8056 | 9.9792 | +0.1736 |
| DNSMOS-P808 | 4.0027 | 4.0279 | +0.0252 |
| DNSMOS-SIG | 3.4828 | 3.4727 | −0.0101 |
| DNSMOS-BAK | 4.1300 | 4.1383 | +0.0083 |
| DNSMOS-OVRL | 3.1579 | 3.1524 | −0.0055 |

The aggregate 8-utterance changes mix six adapter-training utterances with two holdouts. On the two holdouts, SI-SDR improved by `+0.116` and `+0.102 dB`, STOI by `+0.00071` and `+0.00034`, PESQ changed by `+0.0060` and `−0.0360`, and DNSMOS-OVRL by `+0.00026` and `+0.00951`. Two utterances are far too few for a generalization claim. Split results are in `audio_metric_split_summary.json`; aggregate paired bootstrap summaries are descriptive only (`paired_audio_metrics.json`); per-utterance metrics are in `metrics/`. An independent speaker-disjoint CV replication remains required.

## Scope boundary and next gate

This is a rollout-state backend adapter, not a completed Corrective Forcing method. The next valid step is replication on an independent CV/holdout set with a predeclared stop rule. The official test set remains frozen and was not used for tuning.
