# CPEL v2 M/P pilot: execution completed, scientific screen not met

Both authorized arms completed500 matched optimizer updates and all64 final development records. The preregistered incremental candidate criterion was not met: P reduced the raw Tukey paired proxy by1.67035% and its precision-matched counterpart by1.65827% relative to M, below the required10% in both. These are small measured changes, not an assertion of exactly zero effect. Root closes this v2 configuration without A/B expansion, higher pair weights, added steps or window rescue. This outcome does not reject every counterfactual-training approach.

M: PID39967/session52517, exit0, 419.496177s; P: PID30297/session70110, exit0, 327.210369s. Total arm wall time746.706545s. M computed the64-record initial baseline once; P reused its exact hash-bound JSON and did not rerun initial inference. The prior configuration-parser gate failure is preserved in training_gate_attempt01 and before_json_config_parser_fix. The only approved startup correction was JSON parsing; canonical config bytes remain unchanged.

Commands (one scientific execution per arm):

```text
env PATH='/data/coding/sgmse/env/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:/command:/usr/local/cuda/bin' env/bin/python -B experiments/p8_cpel_mp_pilot_20261019/run_pilot.py --execute-arm M --config config.yaml
env PATH='/data/coding/sgmse/env/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:/command:/usr/local/cuda/bin' env/bin/python -B experiments/p8_cpel_mp_pilot_20261019/run_pilot.py --execute-arm P --config config.yaml
```

Frozen canonical config SHA256: `20f750545ff249238dc2e012d474d0ee1f3137abc8218a93cfea9891894864e5`. Config/analysis preparation-time status phrases remain historical snapshots; actual terminal status is recorded in metrics_M.json, metrics_P.json and training_execution_receipt_attempt02.json. No frozen document was edited to alter conditions.

Normal unedited development metrics use all64 fixed files with original frontend/OUVE PC N30. All four metrics are defined for64/64; paired differences are calculated by file then equally averaged.

| Metric | Initial | M | P | M−initial | P−M |
|---|---:|---:|---:|---:|---:|
| normalized_wave_mse | 0.000521001462016 | 0.000518294481143 | 0.000518239479888 | -2.70698087287e-06 | -5.50012547053e-08 |
| si_sdr_db | 5.86471509897 | 5.89312849781 | 5.89284191879 | +0.0284133988417 | -0.000286579020552 |
| stoi | 0.793958822617 | 0.794350142972 | 0.79435471089 | +0.000391320354941 | +4.56791795853e-06 |
| pesq | 1.89985671081 | 1.89502630383 | 1.89516543411 | -0.00483040697873 | +0.000139130279422 |

Control pair score is lower+upper+0.25orth+0.2outside in whitened coordinates. Each control averages its two parent results within a file, then averages files equally. Ratios below are ratios of paired file means, not averages of percentage changes. Whitened outside is distinct from unwhitened outside_change; both are retained separately in aggregate JSON.

| Packet | Matched files/parents | Initial pair mean | M pair mean | P pair mean | P/M | P reduction vs M |
|---|---|---:|---:|---:|---:|---:|
| speech_hann | 64/128 | 4.0695750396e-06 | 3.45012725757e-06 | 3.37302251226e-06 | 0.977651622811 | 2.2348377% |
| speech_tukey | 64/128 | 8.7011573524e-06 | 7.71350532012e-06 | 7.58466298331e-06 | 0.983296525838 | 1.6703474% |
| speech_tukey_pcm16 | 64/128 | 8.73342881736e-06 | 7.74795912326e-06 | 7.6194773448e-06 | 0.983417339144 | 1.6582661% |

Raw Tukey and precision-matched ratios are0.983296525838 and0.983417339144, respectively; both exceed the required≤0.9. M alone improved its proxies relative to initial by about11%, so a comparison solely to initial would misattribute the shared task effect. Per-file normal differences, paired controls and all component means are retained in metrics.json.

| Packet | Label | Evaluated files | Parents | Valid parents | Zero-response parents | Reverse parents | Overgain parents |
|---|---|---:|---:|---:|---:|---:|---:|
| speech_hann | initial | 64 | 128 | 128 | 0 | 0 | 0 |
| speech_hann | M | 64 | 128 | 128 | 0 | 0 | 0 |
| speech_hann | P | 64 | 128 | 128 | 0 | 0 | 0 |
| speech_tukey | initial | 64 | 128 | 128 | 0 | 1 | 0 |
| speech_tukey | M | 64 | 128 | 128 | 0 | 1 | 0 |
| speech_tukey | P | 64 | 128 | 128 | 0 | 1 | 0 |
| speech_tukey_pcm16 | initial | 64 | 128 | 128 | 0 | 1 | 0 |
| speech_tukey_pcm16 | M | 64 | 128 | 128 | 0 | 1 | 0 |
| speech_tukey_pcm16 | P | 64 | 128 | 128 | 0 | 1 | 0 |

The precision-matched packet retains all64 files and both parents, using the same original event center/Gaussian seed and actual quantized clean endpoint difference; no replacement or quantizer bit-depth tuning. Per-endpoint quantizer finite/range/endpoint details, actual target change and branch residual changes remain in each arm JSON. It does not assert exact shared noise after quantization or natural phoneme ground truth.

Noise-instance coverage is identical for initial/M/P:64 codec-engineering-eligible triplets; historical strict43 pass/21 fail remain separately recorded; only37 have sufficient matched edit energy and are evaluated (74 parents). The27 insufficient-energy rows remain non-evaluable and are never zeros or successes. Among the37 evaluated files,30 satisfy strict and7 fail strict. No evaluated noise direction is undefined. Engineering eligibility describes codec-consistent recorded-noise-instance edits, not verified pure noise, true negatives, hallucination rate or FPR.

| Noise group | Matched files/parents | M signed gain mean | P signed gain mean | P−M gain | M change-energy mean | P change-energy mean | P−M energy |
|---|---|---:|---:|---:|---:|---:|---:|
| engineering | 37/74 | 0.0148682118876 | 0.0148131973094 | -5.50145782039e-05 | 1.67092438531e-07 | 1.64834105898e-07 | -2.25833263351e-09 |
| strict_intersection | 30/60 | 0.0160522259529 | 0.0160117813211 | -4.04446317892e-05 | 1.66635331478e-07 | 1.64326564069e-07 | -2.30876740874e-09 |

Signed gain and nonnegative output-change energy are reported together so sign cancellation cannot be mistaken for no response. Processing-chain provenance remains unverified for every triplet; the original strict failures were not silently reclassified or discarded.

Descriptive SNR groups (all four retained; no group chosen as the outcome):

| Bin | Normal files | P−M SI-SDR | P−M STOI | P−M PESQ | Raw pair files / Δmean | Precision pair files / Δmean | Noise files / Δgain |
|---|---:|---:|---:|---:|---|---|---|
| [-10,-5] | 16 | -8.312761719e-05 | -1.551668127e-05 | -6.499886513e-05 | 16 / -2.112485406e-07 | 16 / -2.093108453e-07 | 6 / +7.627609496e-06 |
| [-15,-10) | 16 | +0.001237864835 | +2.030442382e-05 | -0.0002006664872 | 16 / -1.03564e-07 | 16 / -1.042012574e-07 | 8 / -6.136845332e-05 |
| [-18,-15) | 16 | +0.0007502563386 | -4.590551003e-05 | +0.0004323720932 | 16 / -8.440036793e-08 | 16 / -8.388954813e-08 | 12 / -0.0001451474576 |
| [-20,-18) | 16 | -0.003051309638 | +5.938943932e-05 | +0.0003898143768 | 16 / -1.161564388e-07 | 16 / -1.165254631e-07 | 11 / +1.376473334e-05 |

All500 paired step recipes match exactly on step/utt/crop_start/normfactor/center/parent/t/aux_t. Both arms use4 Score forwards,2 autograd.grad traversals and1 Adam/EMA update per step. This is a stop-gradient raw-vector composition rule; the bound concerns current gradients before clip/Adam, not Adam parameter updates or a static-lambda scalar loss. M calculates the same candidate rule but applies0.

| Arm | Updates | Grad traversals | Train Score | Candidate cap steps | Inactive / zero M / zero pair | Applied eta min/median/max | Effective raw ratio min/median/max | Clipped steps |
|---|---:|---:|---:|---:|---|---|---|---:|
| M | 500 | 1000 | 2000 | 35 | 0/0/0 | 0/0/0 | 0/0/0 | 0 |
| P | 500 | 1000 | 2000 | 36 | 0/0/0 | 0.021536129/9.1353018/100 | 0.000424574694/0.01/0.010000000000000002 | 0 |

P maximum0.010000000000000002 is floating-point rounding; scalar budget checks used tolerance1e−12. No nonfinite exception, undefined norm or graph-subset violation occurred. No cap/margin/time/seed/precision retuning was applied. Scalar reconstruction checked all1000 step budgets and exact recipe matching from saved JSON, verified baseline hash reuse and retained immutable source arm JSON hashes. This is bookkeeping validation, not independent GPU gradients/model validation; see training_scalar_validation.json.

| Arm | Total seconds | Normal Score | Controlled Score | Total actual Score | Peak GPU bytes | Backend EMA bytes |
|---|---:|---:|---:|---:|---:|---:|
| M | 419.496177 | 3840 | 1832 | 7672 | 10147200512 | 111187617 |
| P | 327.210369 | 1920 | 916 | 4836 | 10150608384 | 111187617 |

Each terminal backend EMA is saved once without optimizer/frontend/source-checkpoint copies. The initial frontend and intrinsic Fourier W digests match across M/P and remain identical before/after each arm. All704 approved source WAVs and31 other selected current/archive files (735 total) have identical pre/post SHA256. The pre-scan reused the immediately preceding authorized gate protection for unchanged audio, rebound only the reviewed parser/auth changes, and added the archived gate records. No audio was decoded for this summary. No source-data/checkpoint, main source or user-answer writes occurred.

Preservation manifests: training_source_protection_attempt02_pre.json / training_source_protection_attempt02_post.json. Actual terminal handles, commands, timing, input hashes and preservation status: training_execution_receipt_attempt02.json. The older execution_receipt.json/training_execution_receipt.json are preserved historical preflight/gate records. Bridge39452 and server16022 remain alive and untouched. Existing dirty worktree is preserved.

This is a single-seed, existing-CV development screen with related edited controls. It provides no independent content/phoneme/identity or hallucination-reduction confirmation. All numerical changes and missing/non-evaluable coverage are retained; the candidate gate fails rather than being rescued by the normal metrics or selected subgroups. Root has closed this configuration. No new A/B, expansion, weight increase, extra updates or alternate window experiment was started.
