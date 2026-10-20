# Cached continuous-structure readability: completed and independently validated

Unique main run completed in 0.474777s; unique independent validation passed in 5.375103s. Each decoded exactly72 existing PCM files (144 total); main fitted16 fixed dual-output Ridge readouts and independent validator recomputed16 via weighted augmented least squares. This is probe learning, not zero training. Validator verified256 prediction/target arrays; no automatic retry, retuning, new audio, enhancement model, ASR, GPU or network calls.

The frozen 119-dimensional full Noisy representation beat its six-dimensional simple, identity and constant controls on both continuous targets in all four speaker aggregates. Its all-case MSE reduction versus simple Noisy was17.28% for HF energy fraction and8.96% for normalized positive HF magnitude flux. At−18dB the differences versus simple Noisy were small (−0.0018139134 and−0.0000700426). Full frontend was worse than simple frontend on both targets for every speaker. Simple frontend had lower pooled MSE than either full representation. Full Noisy versus full frontend varied across speakers (HF lower for3/4; flux lower for2/4). These are descriptive proxy results; no significance claim or automatic method escalation.

The targets are clean continuous spectral structure proxies, not phoneme labels, content recovery, H1/H0, CAEG, or enhancement benefit. Eight existing exploratory utterances and four speakers do not form a new global holdout; historical/pretraining exposure remains unknown. LOSO constrains only this readout. Frontend uses full-sentence context, so comparisons concern context plus representation readability and cannot isolate information loss.

Commands (each executed once):

```text
/data/coding/sgmse/env/bin/python -B /data/coding/sgmse/experiments/p8_cached_structure_readability_20261019/run_readout.py --execute-after-root-review
/data/coding/sgmse/env/bin/python -B /data/coding/sgmse/experiments/p8_cached_structure_readability_20261019/validate_readout.py --validate-after-completed-run
```

All following MSEs average frames within each case, then average cases equally. Lower is better; predictions remain unclipped. HF=fraction of weighted energy in bins128–240; flux=normalized positive HF magnitude flux.

| Scope | Cases | Arm | HF MSE | Flux MSE |
|---|---:|---|---:|---:|
| all | 32 | full_noisy | 0.0313875850 | 0.0040137652 |
| all | 32 | full_frontend | 0.0346348788 | 0.0041367966 |
| all | 32 | simple_noisy | 0.0379439600 | 0.0044090101 |
| all | 32 | simple_frontend | 0.0275164173 | 0.0034155208 |
| all | 32 | identity_noisy | 0.1456436398 | 0.0158316000 |
| all | 32 | identity_frontend | 0.0320690922 | 0.0049285350 |
| all | 32 | train_mean_constant | 0.0396864897 | 0.0046234309 |
| speaker_p227 | 8 | full_noisy | 0.0361684656 | 0.0037212951 |
| speaker_p227 | 8 | full_frontend | 0.0322381749 | 0.0036324779 |
| speaker_p227 | 8 | simple_noisy | 0.0399142852 | 0.0039313310 |
| speaker_p227 | 8 | simple_frontend | 0.0278530855 | 0.0030746210 |
| speaker_p227 | 8 | identity_noisy | 0.1535622091 | 0.0156715947 |
| speaker_p227 | 8 | identity_frontend | 0.0332401735 | 0.0039629981 |
| speaker_p227 | 8 | train_mean_constant | 0.0423870693 | 0.0042063726 |
| speaker_p228 | 8 | full_noisy | 0.0184609632 | 0.0034228985 |
| speaker_p228 | 8 | full_frontend | 0.0246396677 | 0.0036246917 |
| speaker_p228 | 8 | simple_noisy | 0.0193606340 | 0.0035444145 |
| speaker_p228 | 8 | simple_frontend | 0.0228638698 | 0.0032988770 |
| speaker_p228 | 8 | identity_noisy | 0.1071859455 | 0.0130313723 |
| speaker_p228 | 8 | identity_frontend | 0.0318621236 | 0.0056777372 |
| speaker_p228 | 8 | train_mean_constant | 0.0197312408 | 0.0037359242 |
| speaker_p230 | 8 | full_noisy | 0.0219408263 | 0.0025091229 |
| speaker_p230 | 8 | full_frontend | 0.0273102129 | 0.0029952354 |
| speaker_p230 | 8 | simple_noisy | 0.0248083350 | 0.0028309906 |
| speaker_p230 | 8 | simple_frontend | 0.0203109540 | 0.0024210567 |
| speaker_p230 | 8 | identity_noisy | 0.1501567024 | 0.0161579920 |
| speaker_p230 | 8 | identity_frontend | 0.0264874277 | 0.0040037382 |
| speaker_p230 | 8 | train_mean_constant | 0.0249463216 | 0.0028409759 |
| speaker_p231 | 8 | full_noisy | 0.0489800850 | 0.0064017442 |
| speaker_p231 | 8 | full_frontend | 0.0543514597 | 0.0062947812 |
| speaker_p231 | 8 | simple_noisy | 0.0676925859 | 0.0073293044 |
| speaker_p231 | 8 | simple_frontend | 0.0390377598 | 0.0048675283 |
| speaker_p231 | 8 | identity_noisy | 0.1716697024 | 0.0184654412 |
| speaker_p231 | 8 | identity_frontend | 0.0366866440 | 0.0060696665 |
| speaker_p231 | 8 | train_mean_constant | 0.0716813270 | 0.0077104507 |
| snr_-18 | 16 | full_noisy | 0.0379989369 | 0.0045576272 |
| snr_-18 | 16 | full_frontend | 0.0412830188 | 0.0047780515 |
| snr_-18 | 16 | simple_noisy | 0.0398128503 | 0.0046276699 |
| snr_-18 | 16 | simple_frontend | 0.0362871309 | 0.0043163129 |
| snr_-18 | 16 | identity_noisy | 0.1838220359 | 0.0194494177 |
| snr_-18 | 16 | identity_frontend | 0.0371382454 | 0.0051474834 |
| snr_-18 | 16 | train_mean_constant | 0.0396864897 | 0.0046234309 |
| snr_0 | 16 | full_noisy | 0.0247762331 | 0.0034699031 |
| snr_0 | 16 | full_frontend | 0.0279867388 | 0.0034955416 |
| snr_0 | 16 | simple_noisy | 0.0360750697 | 0.0041903504 |
| snr_0 | 16 | simple_frontend | 0.0187457036 | 0.0025147286 |
| snr_0 | 16 | identity_noisy | 0.1074652438 | 0.0122137824 |
| snr_0 | 16 | identity_frontend | 0.0269999390 | 0.0047095866 |
| snr_0 | 16 | train_mean_constant | 0.0396864897 | 0.0046234309 |

Paired full-minus-control differences (negative favors full):

| Scope | Comparison | HF ΔMSE | Flux ΔMSE |
|---|---|---:|---:|
| all | full_noisy_minus_simple_noisy | -0.0065563750 | -0.0003952449 |
| all | full_noisy_minus_identity_noisy | -0.1142560548 | -0.0118178349 |
| all | full_noisy_minus_train_mean_constant | -0.0082989047 | -0.0006096657 |
| all | full_frontend_minus_simple_frontend | +0.0071184615 | +0.0007212758 |
| all | full_frontend_minus_identity_frontend | +0.0025657866 | -0.0007917384 |
| all | full_frontend_minus_train_mean_constant | -0.0050516109 | -0.0004866343 |
| all | full_noisy_minus_full_frontend | -0.0032472938 | -0.0001230314 |
| all | simple_noisy_minus_simple_frontend | +0.0104275428 | +0.0009934894 |
| speaker_p227 | full_noisy_minus_simple_noisy | -0.0037458196 | -0.0002100358 |
| speaker_p227 | full_noisy_minus_identity_noisy | -0.1173937435 | -0.0119502996 |
| speaker_p227 | full_noisy_minus_train_mean_constant | -0.0062186037 | -0.0004850775 |
| speaker_p227 | full_frontend_minus_simple_frontend | +0.0043850894 | +0.0005578569 |
| speaker_p227 | full_frontend_minus_identity_frontend | -0.0010019987 | -0.0003305202 |
| speaker_p227 | full_frontend_minus_train_mean_constant | -0.0101488945 | -0.0005738947 |
| speaker_p227 | full_noisy_minus_full_frontend | +0.0039302908 | +0.0000888172 |
| speaker_p227 | simple_noisy_minus_simple_frontend | +0.0120611997 | +0.0008567100 |
| speaker_p228 | full_noisy_minus_simple_noisy | -0.0008996708 | -0.0001215160 |
| speaker_p228 | full_noisy_minus_identity_noisy | -0.0887249823 | -0.0096084738 |
| speaker_p228 | full_noisy_minus_train_mean_constant | -0.0012702776 | -0.0003130257 |
| speaker_p228 | full_frontend_minus_simple_frontend | +0.0017757979 | +0.0003258147 |
| speaker_p228 | full_frontend_minus_identity_frontend | -0.0072224559 | -0.0020530455 |
| speaker_p228 | full_frontend_minus_train_mean_constant | +0.0049084269 | -0.0001112325 |
| speaker_p228 | full_noisy_minus_full_frontend | -0.0061787045 | -0.0002017932 |
| speaker_p228 | simple_noisy_minus_simple_frontend | -0.0035032358 | +0.0002455375 |
| speaker_p230 | full_noisy_minus_simple_noisy | -0.0028675087 | -0.0003218677 |
| speaker_p230 | full_noisy_minus_identity_noisy | -0.1282158761 | -0.0136488691 |
| speaker_p230 | full_noisy_minus_train_mean_constant | -0.0030054953 | -0.0003318530 |
| speaker_p230 | full_frontend_minus_simple_frontend | +0.0069992590 | +0.0005741787 |
| speaker_p230 | full_frontend_minus_identity_frontend | +0.0008227853 | -0.0010085027 |
| speaker_p230 | full_frontend_minus_train_mean_constant | +0.0023638913 | +0.0001542596 |
| speaker_p230 | full_noisy_minus_full_frontend | -0.0053693867 | -0.0004861125 |
| speaker_p230 | simple_noisy_minus_simple_frontend | +0.0044973810 | +0.0004099338 |
| speaker_p231 | full_noisy_minus_simple_noisy | -0.0187125009 | -0.0009275603 |
| speaker_p231 | full_noisy_minus_identity_noisy | -0.1226896173 | -0.0120636971 |
| speaker_p231 | full_noisy_minus_train_mean_constant | -0.0227012419 | -0.0013087065 |
| speaker_p231 | full_frontend_minus_simple_frontend | +0.0153136999 | +0.0014272529 |
| speaker_p231 | full_frontend_minus_identity_frontend | +0.0176648157 | +0.0002251147 |
| speaker_p231 | full_frontend_minus_train_mean_constant | -0.0173298673 | -0.0014156695 |
| speaker_p231 | full_noisy_minus_full_frontend | -0.0053713746 | +0.0001069630 |
| speaker_p231 | simple_noisy_minus_simple_frontend | +0.0286548261 | +0.0024617761 |
| snr_-18 | full_noisy_minus_simple_noisy | -0.0018139134 | -0.0000700426 |
| snr_-18 | full_noisy_minus_identity_noisy | -0.1458230990 | -0.0148917905 |
| snr_-18 | full_noisy_minus_train_mean_constant | -0.0016875527 | -0.0000658036 |
| snr_-18 | full_frontend_minus_simple_frontend | +0.0049958878 | +0.0004617386 |
| snr_-18 | full_frontend_minus_identity_frontend | +0.0041447734 | -0.0003694319 |
| snr_-18 | full_frontend_minus_train_mean_constant | +0.0015965291 | +0.0001546207 |
| snr_-18 | full_noisy_minus_full_frontend | -0.0032840819 | -0.0002204243 |
| snr_-18 | simple_noisy_minus_simple_frontend | +0.0035257194 | +0.0003113569 |
| snr_0 | full_noisy_minus_simple_noisy | -0.0112988366 | -0.0007204473 |
| snr_0 | full_noisy_minus_identity_noisy | -0.0826890107 | -0.0087438792 |
| snr_0 | full_noisy_minus_train_mean_constant | -0.0149102566 | -0.0011535277 |
| snr_0 | full_frontend_minus_simple_frontend | +0.0092410352 | +0.0009808130 |
| snr_0 | full_frontend_minus_identity_frontend | +0.0009867998 | -0.0012140450 |
| snr_0 | full_frontend_minus_train_mean_constant | -0.0116997509 | -0.0011278892 |
| snr_0 | full_noisy_minus_full_frontend | -0.0032105057 | -0.0000256385 |
| snr_0 | simple_noisy_minus_simple_frontend | +0.0173293661 | +0.0016756218 |

Out-of-bounds predictions are reported per target; these fractions average cases equally. Frame-weighted fractions are also in execution_receipt.json.

| Arm | HF below0 | HF above1 | Flux below0 | Flux above1 |
|---|---:|---:|---:|---:|
| full_noisy | 0.07223046 | 0.00251920 | 0.08105320 | 0.00000000 |
| full_frontend | 0.32489290 | 0.00009827 | 0.17591415 | 0.00000000 |
| simple_noisy | 0.06478797 | 0.00000000 | 0.03803134 | 0.00000000 |
| simple_frontend | 0.05405453 | 0.00000000 | 0.00019716 | 0.00000000 |
| identity_noisy | 0.00000000 | 0.00000000 | 0.00000000 | 0.00000000 |
| identity_frontend | 0.00000000 | 0.00000000 | 0.00000000 | 0.00000000 |
| train_mean_constant | 0.00000000 | 0.00000000 | 0.00000000 | 0.00000000 |

All32 cases and all full frames were retained (no energy/target filtering). Frame/tail counts below apply to each of the four conditions per utterance; no new PCM decoding was used to write this report.

| Utterance | Frames per condition | Unused tail samples |
|---|---:|---:|
| p227_030 | 329 | 10 |
| p227_031 | 317 | 125 |
| p228_026 | 460 | 0 |
| p228_032 | 418 | 101 |
| p230_027 | 318 | 122 |
| p230_029 | 350 | 51 |
| p231_029 | 310 | 91 |
| p231_032 | 347 | 0 |

Total case-frames: 11396. Full 320-sample frames, hop160, FFT512, fixed symmetric Hann, lambda0.01, weighted train-fold standardization and unpenalized intercept followed the frozen config. No clipping, parameter search or seed expansion.

Protected primary metrics SHA256: `235a7b2ac8f15131d25d1f061319f2d8e1c815d00d53ad08e311192e7c67e913`. Both stages verified72 source hashes before/after and identical code/config hashes; the eight old metrics hashes also match the source preflight. Detailed hashes, complete paired results, fold parameters, heldout predictions, out-of-bound fractions and frame counts are retained in metrics.json, oof_predictions.npz, independent_validation.json and execution_receipt.json. Existing dirty git state is recorded in source_preflight.json and was not changed by experiment edits. Bridge39452 and original listening server16022 remain alive.

Preflight operational failures: the initial handle check matched its own shell text; a subsequent history-command recovery helper recursively selected itself and spawned1798 helper processes. Only that helper descendant tree was stopped. This happened before the scientific run; no scientific script, audio decode, model call or source mutation occurred then. The final execution used direct explicit commands and no history-command recovery.

This bounded experiment is complete. Listening remains a separate optional TODO and did not gate execution. No further experiment, retuning or deployment is authorized by these results.
