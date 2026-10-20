# Native DSM N/D ablation: completed; incremental screen not met

Both authorized native-only arms completed500 paired updates and64 final development records. D reduced raw Tukey and precision-matched weighted pair proxies by3.484244% and3.477313% relative to N. These small positive effects are retained, but both failed the preregistered≥10% improvement criterion. Normal and noise-response damage gates passed; this does not override the failed primary gate. Root closes this specific short-run no-common-point-regression configuration without increasing weights/steps, changing windows/seeds or automatic expansion. This does not establish that the entire counterfactual-training concept fails.

This is the missing native edited DSM→pair contrast: N applies native DSM; D adds the bounded original pair gradient. Common-point clean regression is absent from both, not computed and multiplied by0. N/D are new experiments, not old M/P reruns. Historical M/P and their failed gates are retained unchanged. Neither experiment supplies natural phoneme gold or a full enhancement/content result.

N: PID13601/session12848, exit0, 290.724087s. D: PID54349/session81957, exit0, 290.994855s. Total581.718942s, within900s per arm/1800s total. N completed500/final64/four packets/frozen checks and terminal EMA file SHA before D started. No automatic retry, model preflight, package/network access, ASR, DNSMOS or final-test access occurred.

Unique commands:

```text
env PATH='/data/coding/sgmse/env/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:/command:/usr/local/cuda/bin' env/bin/python -B experiments/p8_cpel_native_pair_ablation_20261020/run_pilot.py --execute-arm N --config config.yaml
env PATH='/data/coding/sgmse/env/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:/command:/usr/local/cuda/bin' env/bin/python -B experiments/p8_cpel_native_pair_ablation_20261020/run_pilot.py --execute-arm D --config config.yaml
```

## Fixed screening order

| Check | Observed | Fixed requirement | Met |
|---|---|---|---|
| Raw Tukey paired proxy | D/N=0.965157555685; 64 files | N mean>1e−12; D/N≤0.9; matched coverage | False |
| Precision matched paired proxy | D/N=0.965226869694; 64 files | N mean>1e−12; D/N≤0.9; matched coverage | False |
| Normal PESQ D−N | +0.00148357264698, n64 | ≥-0.03 | True |
| Normal STOI D−N | +0.000127716480001, n64 | ≥-0.005 | True |
| Normal SI-SDR dB D−N | +0.0395794284526, n64 | ≥-0.2 | True |
| Noise-instance signed input gain D−N | +0.000128925142407, n37 | ≤0.02 | True |
| No zero-response degeneracy | both heldout packets preserve matched nonzero coverage | no trivial zero explanation | True |

## Normal unedited evaluation

All64 normal rows and all four metrics are defined per system. Original source initial64 is explicitly reused from the historical baseline; no new initial inference or old terminal-weight reuse occurred. Raw per-file deltas and four SNR groups are in metrics.json.

| Metric | Source initial | N native only | D native+pair | N−source | D−N | D−source |
|---|---:|---:|---:|---:|---:|---:|
| normalized_wave_mse | 0.000521001462016 | 0.000518464312372 | 0.00051811841631 | -2.53714964414e-06 | -3.45896061565e-07 | -2.8830457057e-06 |
| si_sdr_db | 5.86471509897 | 5.87682764877 | 5.91640707722 | +0.0121125497936 | +0.0395794284526 | +0.0516919782462 |
| stoi | 0.793958822617 | 0.79434181294 | 0.79446952942 | +0.00038299032292 | +0.000127716480001 | +0.000510706802921 |
| pesq | 1.89985671081 | 1.89310566522 | 1.89458923787 | -0.00675104558468 | +0.00148357264698 | -0.0052674729377 |

D−N PESQ+0.00148357/STOI+0.000127716/SI-SDR+0.0395794dB are modest incremental changes. They are not automatically source-relative net deployment improvements; the source→N→D table preserves the actual signs. No favorable checkpoint or development subgroup was selected.

## Controlled responses

Weighted pair score=lower+upper+0.25orth+0.2outside. First average the two parents within a file, then equal files; D/N is the ratio of paired means, not a mean of percentages. Whitened outside is distinct from unwhitened outside_change; both remain separate in JSON.

| Packet | Files/parents | Source mean | N mean | D mean | D/N | Reduction |
|---|---|---:|---:|---:|---:|---:|
| speech_hann | 64/128 | 4.0695750396e-06 | 4.15864580026e-06 | 3.97306547482e-06 | 0.955374818064 | 4.4625182% |
| speech_tukey | 64/128 | 8.7011573524e-06 | 8.80333903413e-06 | 8.49660918405e-06 | 0.965157555685 | 3.4842444% |
| speech_tukey_pcm16 | 64/128 | 8.73342881736e-06 | 8.83509953861e-06 | 8.52787547109e-06 | 0.965226869694 | 3.477313% |

| Packet | Component | N mean | D mean | D−N |
|---|---|---:|---:|---:|
| speech_hann | lower | 8.98013396977e-09 | 8.9624454247e-09 | -1.76885450731e-11 |
| speech_hann | upper | 0 | 0 | +0 |
| speech_hann | orth | 1.5971963408e-05 | 1.52528293541e-05 | -7.19134053973e-07 |
| speech_hann | outside | 7.83374071431e-07 | 7.54478454407e-07 | -2.88956170241e-08 |
| speech_hann | gain | 0.445699745891 | 0.444852735236 | -0.000847010654979 |
| speech_hann | energy | 3.63979152063e-06 | 3.58741667057e-06 | -5.23748500576e-08 |
| speech_hann | outside_change | 5.1962192011e-08 | 5.00455077587e-08 | -1.91668425228e-09 |
| speech_tukey | lower | 2.63947293397e-08 | 2.57211566698e-08 | -6.73572669863e-10 |
| speech_tukey | upper | 0 | 0 | +0 |
| speech_tukey | orth | 3.21993073731e-05 | 3.10531853936e-05 | -1.14612197949e-06 |
| speech_tukey | outside | 3.63558730762e-06 | 3.53795839492e-06 | -9.76289127042e-08 |
| speech_tukey | gain | 0.430134499009 | 0.429531710251 | -0.000602788757533 |
| speech_tukey | energy | 6.42402253009e-06 | 6.33527875767e-06 | -8.87437724209e-08 |
| speech_tukey | outside_change | 2.41153096279e-07 | 2.34677247116e-07 | -6.47584916334e-09 |
| speech_tukey_pcm16 | lower | 2.66109170272e-08 | 2.59310752454e-08 | -6.79841781828e-10 |
| speech_tukey_pcm16 | upper | 0 | 0 | +0 |
| speech_tukey_pcm16 | orth | 3.23249398716e-05 | 3.11765334985e-05 | -1.14840637311e-06 |
| speech_tukey_pcm16 | outside | 3.63626826844e-06 | 3.53905510612e-06 | -9.72131623117e-08 |
| speech_tukey_pcm16 | gain | 0.427856983719 | 0.42728360348 | -0.000573380239075 |
| speech_tukey_pcm16 | energy | 6.42431546918e-06 | 6.33567685474e-06 | -8.86386144344e-08 |
| speech_tukey_pcm16 | outside_change | 2.41198266378e-07 | 2.34749994119e-07 | -6.44827225903e-09 |

| Packet | System | Evaluated files/parents | Valid parents | Zero / reverse / overgain parents |
|---|---|---|---:|---|
| speech_hann | N | 64/128 | 128 | 0/0/0 |
| speech_hann | D | 64/128 | 128 | 0/0/0 |
| speech_tukey | N | 64/128 | 128 | 0/1/0 |
| speech_tukey | D | 64/128 | 128 | 0/1/0 |
| speech_tukey_pcm16 | N | 64/128 | 128 | 0/1/0 |
| speech_tukey_pcm16 | D | 64/128 | 128 | 0/1/0 |

Precision matching retains actual quantized target differences, same raw event location/Gaussian/time/parents and all64 cases. Original per-endpoint range/finite/endpoint flags and target/residual changes remain in the arm JSON; quantization does not assert exact common noise. Same-case precision packets are a countercheck, not independent replication.

Noise-instance codec eligibility64 is distinct from response coverage37:27 insufficient-energy controls are non-evaluable, not zero or successful suppression. Historical strict43pass/21fail remain unchanged; the evaluated37 include30 strict-pass and7 strict-fail. Both parents give74 evaluated engineering responses and60 strict-intersection responses. No evaluated input direction is undefined. Codec compatibility is conditional engineering eligibility, not proven noise purity, true-negative labels or FPR.

| Noise group | Files | N gain mean | D gain mean | D−N signed gain | N energy mean | D energy mean | D−N energy |
|---|---:|---:|---:|---:|---:|---:|---:|
| engineering | 37 | 0.0135398693941 | 0.0136687945365 | +0.000128925142407 | 1.65265594498e-07 | 1.62178962263e-07 | -3.08663223509e-09 |
| strict_intersection | 30 | 0.0146794014009 | 0.0148255712479 | +0.000146169846994 | 1.64446563267e-07 | 1.61377198245e-07 | -3.06936502141e-09 |

Signed gain and nonnegative output-change energy are displayed together; small signed means cannot alone imply no response. All missing/unmatched cases and reasons remain in the64 original rows.

## Recipe, gradient budget and resources

All500 N/D recipes match exactly on step/utt/crop_start/normfactor/event center/parent/native t/aux t. Every step explicitly reports common_clean_aux_computed=False; no common_point import/call or0×aux loss is used. Both arms make4 Score sample forwards and2 autograd.grad traversals per update, then1 Adam and1 backend EMA update. The legacy budget field M_norm is native-only gradient norm here, not old native+common regression.

| Arm | Updates / reverse / train Score | Candidate cap100 steps | Inactive / zero native / zero pair | Applied eta min/median/max | Effective ratio min/median/max | Clipped steps |
|---|---|---:|---|---|---|---:|
| N | 500/1000/2000 | 27 | 0/0/0 | 0/0/0 | 0/0/0 | 0 |
| D | 500/1000/2000 | 28 | 0/0/0 | 0.0186260491/6.46996331/100 | 0.000743399852/0.01/0.010000000000000002 | 0 |

D raw-vector budget≤1% applies before clip/Adam, with1e−12 checking tolerance for floating roundoff (observed maximum0.010000000000000002). It is not an Adam update guarantee, cumulative-trajectory bound or static global-lambda objective derivative. All500 budget rows per arm retained; no graph-None/nonfinite failure occurred. Native/pair norms and cosine distributions are retained in JSON.

| Arm | Seconds | Train Score | Normal Score | Control Score | Peak GPU bytes | Terminal EMA bytes |
|---|---:|---:|---:|---:|---:|---:|
| N | 290.724087 | 2000 | 1920 | 916 | 10144578048 | 111187745 |
| D | 290.994855 | 2000 | 1920 | 916 | 10148510208 | 111187745 |

All frozen frontend states and intrinsic Gaussian Fourier W match before/after and across arms. Terminal backend EMA artifacts were byte-hashed and match their recorded hashes. Only two terminal backend artifacts were saved, no optimizer/raw/source checkpoint copy or audio export.

## Historical context and provenance limits

| Historical auxiliary-regression experiment | Source initial | Historical M | Historical P | P/M pair ratio |
|---|---:|---:|---:|---:|
| speech_tukey | 8.7011573524e-06 | 7.71350532012e-06 | 7.58466298331e-06 | 0.983296525838 |
| speech_tukey_pcm16 | 8.73342881736e-06 | 7.74795912326e-06 | 7.6194773448e-06 | 0.983417339144 |

| Historical system | SI-SDR mean | STOI mean | PESQ mean |
|---|---:|---:|---:|
| historical_M | 5.89312849781 | 0.794350142972 | 1.89502630383 |
| historical_P | 5.89284191879 | 0.79435471089 | 1.89516543411 |

These are historical references on the same exploratory64 CV cases, not four independent arms or a clean causal redundancy interaction. N/D budget denominator is native only while M/P is native+common-point regression; optimizer trajectories and effective coefficients differ. The observations are compatible with different supervision but cannot assign the entire difference to algebraic redundancy. No new A-original-native continued-training control is present. Old M/P failed screening remains unchanged.

Baseline provenance preserves old initial64 SHA and identity, source EMA/checkpoint/dev/seed plus three core source historical protection and exact protected initialization/eval AST equivalence. Added sdes/sampler current locks are not missing historical hashes; exhaustive source-history equivalence is not claimed. No old terminal M/P model was loaded. Source initial was reused as JSON in both arms with zero baseline model calls.

## Preservation, validation and closeout

All737 protected file hashes match pre/post, including704 unique allowed WAVs, checkpoint, code/config/registries/auth/provenance/core/current dependency locks, old baseline/M/P code/results and human exports/current goal. Source audio was only byte-hashed for preservation; report generation performed no new PCM/model processing. Compact manifests are source_protection_pre.json/post.json. Dirty git state is retained in execution_receipt.json; no frozen source/result, Root document or user answer was modified.

Saved-JSON validation independently reconstructed all64 paired normal deltas and per-parent component sums/paired ratios from original arm records, and checked1000 step budgets and500 recipe matches. It is scalar bookkeeping, not an independent GPU gradient/model or acoustic experiment. Source arm JSON hashes are unchanged. Scalar results and full per-file/four-SNR descriptions are in metrics.json/independent_validation.json.

The initial metadata precheck helper assumed local registries, then correctly followed the frozen config paths before any scientific command. This metadata-only correction and actual PID/session/exit sequence are in execution_receipt.json; no scientific run was retried. Compatibility bridge39452 and server16022 remain alive and untouched.

The N/D configuration is closed by its fixed incremental screen. Small positive effects are acknowledged without a phoneme/content/identity or hallucination-reduction claim. No expansion, alternate seed, greater weight, added steps or window rescue was started. Long-term research scope remains broader than this bounded comparison; the result is not a global blocker or proof that the original concept can never work.


Descriptive fixed SNR bins (no group selection):

| Bin | Normal n | D−N SI-SDR | D−N STOI | D−N PESQ | Raw N→D means | Precision N→D means | Noise files |
|---|---:|---:|---:|---:|---|---|---:|
| [-10,-5] | 16 | +0.00400519868 | +2.09954275e-05 | +8.0190599e-05 | 1.35872154e-05→1.30505083e-05 | 1.36008926e-05→1.30639593e-05 | 6 |
| [-15,-10) | 16 | +0.0041465006 | +0.00030112526 | +0.00360416621 | 4.46753789e-06→4.2292799e-06 | 4.48207567e-06→4.2429983e-06 | 8 |
| [-18,-15) | 16 | +0.0054751445 | -5.14497675e-05 | -2.8885901e-05 | 4.85534376e-06→4.67021e-06 | 4.92118693e-06→4.73553918e-06 | 12 |
| [-20,-18) | 16 | +0.14469087 | +0.000240195 | +0.00227881968 | 1.2303259e-05→1.20364386e-05 | 1.23362429e-05→1.20690051e-05 | 11 |
