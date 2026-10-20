# 3321 original condition-gap mechanism probe: attempt02

Status: completed

Oracle clean-tail intervention is an OOD mechanism probe, not a deployable method. This is a preidentified picked exploratory positive case on a second utterance, not unbiased population proof or a trained method. Prior P8-M 5243 active masking negative is preserved and is a distinct experiment.

Actual execution stages: {"backend_runs_completed": 2, "config_validated": true, "frontend_completed": true, "model_loaded": true}

Completed two off runs with NFE30 after exact frontend and no_op PCM16 gates. Window361920:378560, internal320-sample fades, original float frontend held unchanged. Raw comparisons use new no_op raw; PCM16 is reference only. Fixed clean activity threshold−70dBFS on20ms frames.

|Run|Tail RMS dBFS|4–7.5kHz mean square|Outside relative L2|Active relative L2 vs no_op|
|---|---:|---:|---:|---:|
|no_op|-59.04300|1.7817435e-09|0|0|
|condition_tail_clean|-88.79402|1.4456853e-11|0.015713464|0.014069088|

## Worker cached verification

Both frontend and no_op replay PCM16 file hashes match the original3321 outputs. All source, checkpoint, model-code, config, driver and raw/PCM output hashes matched. The globally held frontend and exact condition interpolation were verified; condition samples outside361920:378560 and the normalization peak are unchanged. Both runs used NFE30. Independent raw/PCM RMS and SciPy boxcar-periodogram highband energies matched saved metrics. No inference was repeated.

Source basename nominal SNR is−19dB. Direct full clean/archived-noise SNR is -18.99974dB; fixed-gap ratio is -63.88376dB. Gap clean RMS 2.51307207e-05, archived noise RMS 0.0393000579, actual noisy RMS 0.0393001202. Archived noise is provenance/diagnostic only and did not construct model inputs.

| Representation | Run | Gap RMS | Gap dBFS | RMS change dB | Highband energy | Highband change dB | Outside L2 | Active L2 vs no_op | Active L2 vs clean |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| raw | no_op | 0.001116477 | -59.043 | 0.000 | 1.78174347e-09 | 0.000 | 0.000000 | 0.000000 | 0.610313 |
| PCM16 | no_op | 0.00111668238 | -59.041 | 0.000 | 1.81615356e-09 | -0.000 | 0.000000 | 0.000000 | 0.610333 |
| raw | condition_tail_clean | 3.63328231e-05 | -88.794 | -29.751 | 1.44568525e-11 | -20.908 | 0.015713 | 0.014069 | 0.611415 |
| PCM16 | condition_tail_clean | 4.37143719e-05 | -87.188 | -28.146 | 3.44765753e-11 | -17.216 | 0.016259 | 0.014553 | 0.611423 |

The original L2 uses float32 reference norms; float64 denominators differ at most 4.8970309e-05 relative. Original precision reproduces the recorded values. The table uses float64 denominators.

This corrected attempt follows a preserved pre-model configuration failure with zero runs. This is a preselected exploratory second utterance, not independent confirmation or an unbiased estimate. Energy changes do not establish speech correctness, deployability or novelty.

The3321 raw gap reductions (29.751dB RMS,20.908dB highband) have the same direction as10720 condition interventions at10dB attenuation (25.657/44.902dB) and20dB attenuation (29.372/37.573dB). The highband magnitudes differ.3321 full active-clean L2 changes from0.610313 to0.611415, a small increase despite local gap suppression; the local energy result is not a global quality improvement.
# Research Lead interpretation

At a second, exploration-selected utterance, local condition replacement
again strongly suppresses gap energy while the full frontend, common random
numbers, input outside the gap and normalization peak are fixed. This extends
the10720 path-sensitivity observation beyond that one waveform. It remains
an oracle, off-distribution intervention on a picked example, not independent
confirmation or a deployable controller.

The active-clean distance rises slightly (float64 recomputation0.610313 to
0.611415). Consequently the experiment does not establish a quality or content
improvement. A future mechanism must protect real weak consonants and show
incremental value over simple noise/energy controls before training is justified.
Keep the5243 d human positive candidate and known-identity VCV controls.

The original sibling-directory attempt failed before model load because of
YAML1.1 parsing of unquoted off as False. No inference was executed there;
that failure is retained. This corrected attempt used explicit string/boolean
validation and passed exact frontend and backend no-op gates.

