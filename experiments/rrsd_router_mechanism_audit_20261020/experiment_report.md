# RRSD Router mechanism audit (2026-10-20)

## Purpose

This read-only audit checks that the mechanism described in the thesis draft matches the actual 80k checkpoint and the current source code. It does not load the DNS dataset, regenerate audio, modify the checkpoint, or start training.

## Verified checkpoint facts

- Checkpoint: `joint_train_logs/sb61i0gj_RRSD/epoch=118-step=80000.ckpt`
- `global_step=80000`, `sde=ouve`, `N=30`
- `rrsd_teacher=future_grad`
- `rrsd_freeze_base=True`
- `rrsd_weight=1.0`, `rrsd_future_weight=1.0`
- `rrsd_smooth_weight=0.01`, `rrsd_future_gate_cost=0.001`
- `rrsd_use_gate=False`
- `contrastive_weight=0.0`
- State dict contains 8 `rrsd_router.*` tensors, 42,577 Router parameters, and 55,118,388 non-Router parameters.

## Source-code correspondence

`_rrsd_future_grad_loss` constructs routed and alpha-zero counterfactual next states at the same future time, evaluates both through one stacked additional frozen Score forward, converts each to `x0`, and forms a relative future error with a detached base denominator. The function also contains optional perceptual-conflict and routed contrastive paths; both are disabled in the audited checkpoint (`rrsd_perceptual_conflict_weight` absent/zero and `contrastive_weight=0`).

## Conclusion

The thesis mechanism description is consistent with the audited checkpoint: the reported main result is a future-gradient Router experiment, not a WavLM/UniSpeech contrastive-loss experiment and not an explicit gate experiment.

Hashes, exact hyperparameters, tensor counts and checks are in `metrics.json`.

## Execution note

The first invocation exposed a path-resolution issue when the script was launched from its experiment directory: checkpoint unpickling could not find the repository's `sgmse` package. The script now inserts the repository root into `sys.path`; this was an audit-script issue, not a checkpoint or dataset failure.

## Reproduction

```bash
/data/coding/sgmse/env/bin/python experiments/rrsd_router_mechanism_audit_20261020/audit.py
```
