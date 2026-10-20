# CPEL attempt03: integration passed; dev additivity premise failed

Unique separately Root-authorized attempt completed with exit0, PID37885, exec session51647 (completed), script time17.186259s. The corrected JointWaveScpDataModule loader allowed strict source-checkpoint construction and one source EMA materialization. No fallback, install, additional attempt or scientific-condition change followed.

```text
env PATH='/data/coding/sgmse/env/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:/command:/usr/local/cuda/bin' /data/coding/sgmse/env/bin/python -B /data/coding/sgmse/experiments/p8_cpel_mp_pilot_20261019/preflight_attempt03/preflight_pilot.py --execute-preflight
```

Exact PATH and existing Ninja path/version are in execution_receipt.json. CLI output was `Running NCSN++ as discriminative backbone` and `[EMA] checkpoint EMA loaded successfully, shadow params = 8`. Extension-cache reuse versus rebuild was not separately instrumented. Four frozen Python hashes, configs and registries match dedicated authorization; earlier attempts remain preserved.

Actual execution:6 permitted PCM reads (clean/noisy/noise for train[0] and dev[0]),2 native Score+2 common-aux Score+30 normal-dev Score forwards,2 backward traversals,0 optimizer steps,0 EMA updates and0 parameter updates. No weights/audio were saved, no network attempt occurred. Peak CUDA allocated memory was9805033984 bytes. The source checkpoint,6 WAVs and4 drivers have identical pre/post hashes.

Additivity premise, fixed center crops, tolerance relative RMS≤0.001 (no lag/gain repair or reselection):

| Crop | Utterance | Relative RMS error | Premise |
|---|---|---:|---|
| train | reader_05264_fileid_4279 | 0.000193562431377 | passed |
| dev | reader_11236_fileid_6794 | 0.00335558806546 | failed |

The script's passed status means bounded model/gradient/sampler integration completed. It does not mean both additivity checks passed. In particular, the dev crop cannot be treated as a validated pure-noise intervention or included as a successful false-positive/noise-selectivity control. No additivity or content claim is extended to the other registry records.

Single train-crop losses: native=0.00611720886081, common-point auxiliary=0.00611567310989, pair=1.53820806759e-07, total=0.0122328978032. Pair active=true; response=0.188469856977, lower=0.0, upper=0.0, orth=5.52566120859e-07, outside=7.83963969297e-08. These are phase-aware compressed-spectrum proxy values, not phoneme recovery or suppression results.

Single-crop gradient scale:

| Quantity | L2 norm / ratio |
|---|---:|
| pair gradient | 3.41251333026e-06 |
| M gradient reconstructed as total−0.1pair | 0.0322193779343 |
| total P gradient | 0.0322193988597 |
| effective 0.1×pair/M gradient ratio | 1.05914935329e-05 |

All271 dnn tensors with gradients were finite; no non-dnn gradient was observed. Detached clean/noisy/frontend/common-state controls passed. This is one crop/time/parent and not a population gradient estimate; no weight was changed to amplify it, and no training benefit is inferred.

Normal-dev inference completed exactly30 Score calls/reported NFE30 and produced32640 finite samples, peak1.00275826454. This verifies geometry/finiteness only; no output audio or objective quality/content evaluation was saved or claimed. Peak above1 is retained as observed, with no clipping.

Model parameter/buffer pre/post digest both `533d10eca54f5ff711f1697768d3e34b557d9cb06873239beceef325cd7f99a4`. Thus the preflight did not update the loaded model. All14 protected earlier-attempt artifacts retain their preflight hashes. This attempt's source/code/config hashes, complete command/PATH/Ninja information, actual PID/session/timing/counts and results are in ../execution_receipt.json. Compatibility bridge39452 and server16022 remain alive and untouched. No Root plan/evidence or main source was edited.

This subdirectory contains dedicated config and original result byte copies plus this report; metrics were not recomputed. Parent M/P training is unexecuted/incomplete and root_execution_authorization.json remains absent. Integration preflight is stopped; future M/P execution requires separate Root authorization and must preserve the failed dev-additivity limitation.
