# Preserve native fixed Fourier frequencies — preparation-only correction

Root identified a scientific-semantic bug after attempt03 passed integration: the prepared all-`dnn.*` unfreeze also enabled native fixed Fourier frequencies. `sgmse/backbones/ncsnpp_v2.py:681` constructs `layerspp.GaussianFourierProjection`; `ncsnpp_utils/layerspp.py:37` defines its `W` as `nn.Parameter(..., requires_grad=False)`. It is not an ordinary trainable backend weight.

Only the two prepared entry sources were changed. Their previous bytes/SHA are preserved under `before_intrinsic_fixed_fix/sha256_manifest.json`. No attempt01/02/03 snapshot, result, source checkpoint, main SGMSE code, loss, edit, sampler or numerical hyperparameter was changed.

`run_pilot.enable_backend_training` now restores backend learnability, identifies native `GaussianFourierProjection` modules by actual module type, and re-freezes each direct `W`. It records full excluded parameter names and checks that these are the only frozen backend exceptions; non-backend parameters remain frozen. For the source's native Fourier embedding placement the expected excluded name is **`dnn.all_modules.0.W`**. Actual runtime exclusion names will be recorded by a future authorized execution, not invented here.

- Optimizer and preflight `autograd.grad` lists include only `requires_grad=True` parameters.
- Preflight rejects any populated gradient on a frozen parameter as well as non-dnn/nonfinite gradients.
- New backend EMA allocates, updates and copies back only the trainable parameters. Fixed Fourier `W` is never multiplied by .999/.001, preventing avoidable rounding drift.
- Final backend `state_dict` still includes unchanged fixed `W` and all buffers; no key/topology is removed.

One tiny synthetic CPU example used a two-frequency fixed-W module, a learnable linear layer, a frozen dummy frontend and a fixed buffer. After a synthetic optimizer step and trainable-only EMA update/copyback, fixed W/buffer state_dict bytes remained tensor-exact, W/frontend had no gradients, and the exclusion list matched the dummy module. Both edited Python ASTs passed. No native model, weight file, WAV/PCM, CUDA, fourth preflight or500-step run was invoked.

**Attempt03's271 gradient tensors describe the old all-unfrozen collection.** Its successful integration and zero-update evidence remain unchanged; they do not verify the corrected parameter collection or the revised EMA behavior on the actual source model. Root has not authorized a fourth preflight or M/P training. Current old authorization hashes are stale after this correction and must not be silently reused.
