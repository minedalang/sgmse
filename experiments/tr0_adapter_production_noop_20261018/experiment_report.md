# TR-0 production adapter exact no-op — 2026-10-18

## Status

Completed with the optional adapter path implemented in `sgmse/backbones/ncsnpp_v2.py`. No training, dataset inference, checkpoint write, or original model state modification was performed.

## Protocol

The locked 80k NCSN++ v2 checkpoint is loaded twice with identical model configuration. One model uses the unchanged default path. The second enables `backend_adapter_enable=True` at `all_modules.33`, the audited 128-channel late decoder block. The adapter is a separate zero-initialized 1×1 residual projection, so original `all_modules` indices and checkpoint parameters remain unchanged.

The original state dict loads strictly into the default model. The enabled model loads it with only the two expected missing adapter parameters; these are initialized to zero by construction.

The complete `JointFrontendSgmseModel` also loaded successfully from the same checkpoint with `weights_only=False` (the trusted local checkpoint contains a serialized datamodule object that PyTorch's newer `weights_only=True` default rejects). The loaded default model reports `backend_adapter_enable=False` and has no adapter module.

## Result

On the same deterministic complex input and timestep, the complete outputs were exactly equal:

- `max_abs_diff = 0`
- `rmse = 0`
- `torch.equal = true`

The adapter contains 16,512 parameters. This proves the optional production path preserves the baseline at initialization. It does not show that the adapter can train or improve enhancement quality.

## Safety boundary

The adapter is disabled by default, and the original checkpoint was not overwritten. A future training script must explicitly load the adapter-enabled configuration with non-strict base loading and save its adapter checkpoint separately.
