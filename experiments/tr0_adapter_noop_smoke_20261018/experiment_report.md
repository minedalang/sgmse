# TR-0 adapter exact no-op smoke — 2026-10-18

## Status

Completed on CPU without training, inference over dataset audio, or checkpoint modification. The existing NCSN++ v2 is loaded from the locked 80k checkpoint. A candidate zero-initialized complex residual adapter is applied outside the production model solely to test the no-op contract.

## Result

The external candidate is a 1×1 real/imag residual projection with six parameters. Both weight and bias are zero initialized. On a deterministic random complex input with the checkpoint's actual model dimensions, the wrapped output is exactly equal to the base output (`max_abs_diff=0`, `rmse=0`, `torch.equal=True`).

I also simulated the audited late decoder insertion at `dnn.all_modules.33` by wrapping that block with a zero-initialized feature residual projection. The feature adapter has 16,512 parameters (128-channel 1×1 convolution), and the complete wrapped NCSN++ output again matched the unwrapped output exactly (`max_abs_diff=0`, `rmse=0`, `torch.equal=True`).

This establishes only that a zero-initialized wrapper can preserve the checkpoint at initialization. It does not show that an adapter improves enhancement quality, and it is not yet a production insertion or training result.

## Boundary

The smoke uses an in-memory wrapper rather than modifying the NCSN++ decoder or checkpoint. It proves the no-op property at the candidate module but does not prove trainability, quality improvement, or the correctness of a persistent production integration. The original checkpoint and dataset remain untouched.
