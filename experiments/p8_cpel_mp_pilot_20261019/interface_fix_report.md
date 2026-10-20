# Targeted static interface repair after two terminal preflight failures

**No third real preflight, model instance, checkpoint storage access, WAV header/PCM, installation or scientific call was performed for this repair.** Attempts01/02, their snapshots, original results and source project code are untouched. The prior four prepared Python files were copied verbatim into `before_interface_fix/`, with SHA256/byte manifest.

## Exact cause and minimal repair

`joint_train.py:162–165` selects `JointWaveScpDataModule` for `joint_scp`. The source metadata has non-null `train_wav_scp`, `valid_wav_scp`, `clean_dir` and no `base_dir`. The latest effective classes in `sgmse/joint_data_module.py` are the second definitions, not the earlier duplicates:

- final `JointWaveDataModule` line629 / constructor651 requires `base_dir`;
- final `JointWaveScpDataModule` line673 / constructor700 requires `train_wav_scp`, `valid_wav_scp`, `clean_dir`.

The prepared runner/preflight had explicitly overridden the source DM with the wrong base-directory class. Old successful `joint_enhance_by_dir.py:244–258` uses that class **together with** an explicit `base_dir='.'` inference placeholder. Copying only its class without its companion arguments caused attempt02. This repair does **not** add the fake path; both prepared entry points now use the actual training mapping, `JointWaveScpDataModule`, with the source's saved SCP constructor arguments. `strict=True`, source topology, EMA handling and every neural computation/loss/edit/sampling choice are unchanged.

The authoritative existing JSON fields are at `experiments/p8_checkpoint_base_identity_20261019/metrics.json` → **`comparison.A_hparams`**, not the top level or `metadata` (the latter is a list of metadata receipts). `dataset` was not saved as a useful value, and `data_module_cls` is represented as an inert type; the class-name claim therefore comes from current training mapping plus the saved required SCP arguments, not from inventing a recovered class name. Their old absolute SCP locations need not exist: final DM constructor only stores the strings; neither entry point calls DM `setup()` or any dataloader. Installed Lightning `core/saving.py:_load_state` merges saved hparams with explicit overrides, constructs the model, then loads state; it does not call dataset setup for this module.

## Whole affected interface check

| Prepared use | Actual source definition / static finding |
|---|---|
| `JointSemambaSgmseModel` | `joint_model.py:5888` alias to `JointFrontendSgmseModel`; unchanged strict loader |
| `model.sr` | assigned in `model.py:151`; saved source16000. Not incorrectly requested from DM |
| `dm.n_fft/hop_length/num_frames/window` | `_BaseJointWaveDataModule.__init__:306–335`; saved510/128/256/hann, crop32640 |
| `dm.transform_type/spec_factor/spec_abs_exponent/normalize` | same base constructor; saved exponent/.15/.5/noisy |
| `_sgmse_stft([1,L])` | `joint_model.py:1832–1855`: local `dm.window.to(wav.device)`, returns complex `[B,1,F,T]` after `dm.spec_fwd` |
| `frontend.forward_transformed_spec` | `frontends/storm.py:195–242`: accepts same compressed complex four-dimensional input, pads/crops frame axis internally |
| `dm.spec_fwd/spec_back` | base lines345–378, inherited unchanged by both DM variants |
| `to_audio(sample.squeeze(),length)` | `model.py:1364–1393`→base DM `spec_back`/`istft`; accepts `[F,T]`, length32640; `istft` moves its local window to spec device |
| `get_pc_sampler` | `model.py:1296–1314` copies SDE, sets N30, forwards explicit original score function; none-corrector has n_steps0 (`correctors.py:85–90`) |
| native Score callable | sampler invokes `(state,anchor,time)`; closure supplies original Noisy via `y_cond`, preserving mode2 source inputs |
| `eval(no_ema=False)` then `to(device)` | `model.py:424–448` consumes pending source EMA and materializes once. `to:1289–1294` only transfers existing EMA tensors, helper if any, and module; it does not re-copy EMA over dnn or rebuild parameter tracking |
| backend unfreeze | source EMA is materialized while original frozen topology/parameter set remains intact; afterwards only dnn parameters are trainable. No later full-model train/eval call; F and dnn `.eval()` themselves do not replace weights or disable autograd |
| device/window | DM plain window need not migrate with model; every called STFT/ISTFT explicitly copies it to the input device. F has its own registered nonpersistent window and uses local device copy |

Saved supplemental fields are backbone=`ncsnpp_v2`, sr16000, t_eps.03, normalize=noisy; `gpu`/`gpus` are not saved. The ScoreModel default `kwargs.get('gpus',0)>0` thus has a defined boolean and no duplicate saved `gpu` argument. Saved contrastive/content weights remain zero and frontend_ckpt null, as previously checked. No new external initializer or helper model was introduced.

## Static verification and remaining risk

One bounded AST check selected the **last** class definitions, extracted effective constructor required arguments, checked all required SCP keys in the saved existing JSON, verified both corrected callsites, and parsed all four prepared Python files. Passed. No DM or model was instantiated. This checks interface consistency, not actual strict loading or GPU operation.

Attempt01's missing project-env PATH remains an execution-environment obligation; the binary's existence does not itself repair subprocess lookup. Root/Worker must retain the separately recorded PATH correction for any future authorized attempt. Current parent authorization contains old code hashes and must not be silently reused; stale hashes should fail. The old start marker/results are terminal evidence and have not been removed here. Root alone decides whether to authorize another isolated attempt.

Real strict checkpoint load, EMA pending-state compatibility, source tensor/device/dtype behavior, CUDA deterministic kernels, gradient memory/runtime, waveform-source integrity/additivity and actual30-call inference are **still unverified by this static repair**. These are runtime risks, not asserted success. No base_dir workaround, strictFalse, fallback topology, package install, sampler/loss change or third execution was used.
