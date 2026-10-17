# TR-0 adapter-only gradient smoke — 2026-10-18

The optional production adapter at `all_modules.33` was initialized from the locked 80k checkpoint. All NCSN++ parameters except the adapter were frozen. A single synthetic tensor loss was backpropagated only to check gradient routing; this is not a training result and uses no dataset audio.

Results:

- Adapter parameters: 16,512
- Adapter gradients: present, finite, nonzero
- Frozen base parameters: all `requires_grad=False`, all gradients absent
- Original checkpoint and dataset: unchanged

This clears the implementation gate for a future short screen. It does not establish a useful enhancement objective or any quality gain.
