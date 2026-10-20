# P7-A probability-flow ODE convergence smoke — 2026-10-17

**Status: incomplete; stopped at a compute-cost gate.** This is a partial technical smoke, not a completed convergence study or speech-quality evaluation. No DNSMOS migration/evaluation or training ran in this experiment.

## Objective and fixed setup

The planned test compares SciPy RK45 probability-flow ODE integrations at `rtol=atol` of `1e-3`, `1e-5`, and `1e-7`, using the same saved OUVE prior draw per utterance. It uses the fixed P7-0 80k checkpoint (`epoch=118-step=80000.ckpt`), the first four files in the existing P7 baseline manifest, the checkpoint-loaded StoRM frontend, `model.eval(no_ema=False)`, and `denoise=False`. It is a solver-sensitivity diagnostic only; the 32-item development split has already been used for screening.

Checkpoint and manifest hashes matched the configuration. The runner verified each input hash before inference. The first utterance is a 30-second, 16 kHz file. No source data or checkpoint was modified.

## Execution and observed results

The first launch failed before model loading because the shell `PATH` omitted the existing SGMSE environment's `bin` directory. Ninja was already installed in `/data/coding/sgmse/env` (version 1.13.2); no package was installed. The run succeeded after adding `/data/coding/sgmse/env/bin` to `PATH`.

| Utterance | `rtol=atol` | Result | SciPy `nfev` | Score forward hooks | Elapsed |
|---|---:|---|---:|---:|---:|
| `000_fileid_2673.wav` | `1e-3` | Completed; finite output | 26 | 26 | 15.74 s |
| `000_fileid_2673.wav` | `1e-5` | Completed; finite output | 92 | 92 | 45.55 s |
| `000_fileid_2673.wav` | `1e-7` | Interrupted with SIGINT after about 14.5 minutes in that solve; no endpoint or NFE returned | — | — | >14 min |
| Remaining three manifest files | all tolerances | Not run | — | — | — |

The saved initial state SHA256 is `385d538a61c2d30b7d8e2d53ef1fed102ccfac7c41361f545a4b1eb1afe36058`. Completed WAV hashes are in `metrics.json`. The two completed runs use that same initial state; their SciPy NFE values equal independently hooked Score-network forward counts.

As a post-hoc check on the two saved FLOAT WAVs, the relative waveform L2 difference (`1e-3` against `1e-5`) is `0.0756460`, maximum absolute sample difference is `0.0105964`, and the relative L2 difference between their complex STFTs is `0.0756448` using a periodic Hann window, `n_fft=510`, and `hop_length=128`. Both WAVs are finite, 480,000 samples long, and have peak magnitude below `0.029`, so WAV clipping did not affect this comparison. The difference is far above the planned `1e-3` convergence threshold. It does **not** establish which tolerance is accurate: the `1e-7` reference did not finish.

The `1e-7` solve remained actively compute-bound on the L40S and reached the pre-set 15-minute observation limit without returning. The run was stopped to avoid letting one smoke solve consume an unbounded GPU allocation. There is no evidence of a numerical exception or a frozen process; this is a cost finding, not a solver failure.

## Conclusion and next decision

The original three-tolerance, four-utterance design was not demonstrated feasible: on the first 30-second sample, `1e-5` costs about 46 seconds and the stricter `1e-7` run exceeded 14 minutes without a result. The bounded `1e-6` follow-up completed in 174.03 seconds, but still differs from the `1e-5` output by about `0.00399` relative L2 in waveform and complex STFT, above the planned `1e-3` threshold. Taken together, these measurements do not establish numerical convergence or identify an accurate tolerance; P7-A remains incomplete and further tolerance runs are stopped pending a precision-path review.

## ODE precision-path audit (static plus isolated SciPy check)

The shared implementation in `sgmse/sampling/__init__.py::get_ode_sampler` reconstructs every SciPy state with `from_flattened_numpy(...).to(device).type(torch.complex64)` inside `ode_func`, then returns the drift to SciPy through `to_flattened_numpy`. The endpoint is also explicitly cast to `complex64`. With the installed SciPy 1.15.3, an isolated CPU `solve_ivp` check confirmed that a `complex64` initial vector is promoted to `complex128` for the solver's internal state; however, the repository callback re-quantizes each intermediate state to `complex64` before evaluating the model drift. Thus the adaptive solver's state arithmetic is higher precision than the callback's state/model path. This is a plausible precision floor and a reason tighter tolerances may increase function calls without proportionate output improvement, but the floor and its contribution to the observed deltas have not been measured. No shared sampler code was changed, and this audit does not establish that `complex64` caused the observed sensitivity.

Decision: do not run more strict-tolerance GPU integrations from the original matrix. First decide whether a separate, bounded precision-path experiment is warranted; any such experiment must preserve the fixed checkpoint, same `x_T`, independent Score-call count, and be explicit about whether it changes the model-evaluation precision. No speech-quality metrics or DNSMOS are needed to answer the current numerical-method question, and this ODE diagnostic is separate from the DNSMOS GPU migration.

Do not launch the remaining 11 solves from the original design. The practical-tolerance point above applies to only this utterance; any future extension needs a separately bounded scope and must preserve the same-initial-state and independent NFE checks. Any shorter-segment result must be labeled as such and cannot be generalized to full-utterance convergence. No speech quality metrics were computed; no DNSMOS values were generated.

## Reproducibility artifacts

- `config.yaml`: planned conditions plus actual partial-run status.
- `metrics.json`: checkpoint/input provenance, completed results, post-hoc pairwise deltas, interruption, and unrun cases.
- `run.log`: first PATH failure and the successful partial run ending in SIGINT.
- `shared_xT/000_fileid_2673.pt` and the completed FLOAT WAVs are retained in this directory.
- Original runner command with the environment PATH correction:

```bash
cd /data/coding/sgmse && PATH=/data/coding/sgmse/env/bin:$PATH /data/coding/sgmse/env/bin/python experiments/storm_error_attribution_p7_20261017/ode_convergence_smoke_20261017/run_ode_convergence_smoke.py --device cuda
```

## Bounded practical-tolerance follow-up (2026-10-17)

After the `1e-7` cost stop, a single additional run was completed for `000_fileid_2673.wav` at RK45 `rtol=atol=1e-6`, with `denoise=False`, the fixed 80k checkpoint, and the previously saved `shared_xT/000_fileid_2673.pt`. The runner reused the saved state without writing it and compared against the existing `1e-5` FLOAT WAV. No other utterance or tolerance was run. One launch attempt exited before model loading because the standalone runner had not added the repository root to `sys.path` (`ModuleNotFoundError: sgmse`); this was corrected before the successful run, with no integration or output from the failed attempt.

| Utterance | `rtol=atol` | Result | SciPy `nfev` | Score forward hooks | Solve time | Total time |
|---|---:|---|---:|---:|---:|---:|
| `000_fileid_2673.wav` | `1e-6` | Completed; finite endpoint and waveform | 464 | 464 | 174.03 s | 183.02 s |

The 30-second output contains 480,000 samples at 16 kHz, has peak magnitude `0.02836`, and has zero clipped samples. Relative to the saved `1e-5` WAV, waveform relative L2 is `0.00398533`; relative L2 between reconstructed complex STFTs is `0.00398575` using the checkpoint's `n_fft=510`, `hop_length=128`, and periodic Hann analysis window. Both remain above the planned `1e-3` comparison threshold. This single pair shows that `1e-6` is materially closer to `1e-5` than the prior `1e-3`/`1e-5` pair, but it does not establish numerical convergence or identify either result as accurate.

The checkpoint hash, P7 manifest hash, input hash, and saved `x_T` hash matched the existing record. Post-run checks confirmed that `metrics.json`, the saved `x_T`, and the existing `1e-5` WAV hashes were unchanged. The new 1e-6 WAV and its independent provenance/runtime/delta record are `rk45_rtol_atol_1e-06/000_fileid_2673.wav` and `supplemental_metrics_1e-6.json`; the reproducible bounded runner is `run_practical_1e6.py`. No DNSMOS, speech-quality metrics, training, or additional solves ran. The 1e-7 stop and the unrun utterances remain as recorded above.

Command:

```bash
cd /data/coding/sgmse && PATH=/data/coding/sgmse/env/bin:$PATH /data/coding/sgmse/env/bin/python -u experiments/storm_error_attribution_p7_20261017/ode_convergence_smoke_20261017/run_practical_1e6.py
```
