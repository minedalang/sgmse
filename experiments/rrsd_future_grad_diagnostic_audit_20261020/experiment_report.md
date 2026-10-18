# Archived Future-gradient diagnostic audit (2026-10-20)

## Scope

This audit parses the existing `future_gd_smooth.log`; it does not repeat the training command. The source run used `SGMSE_RRSD_DIAG_BATCHES=16`, finite-difference probing, Smooth Oracle probing, and an explicit diagnostic exit without optimizer save.

## Verified log evidence

- All diagnostic batches `1/16` through `16/16` are present.
- Every batch reports a positive absolute future-gradient magnitude (`grad_abs`), so the archived run exercised a differentiable future signal rather than a permanently empty diagnostic.
- There are 16 finite-difference records with `eps=0.01`; both zero-route and positive-route deltas are recorded across the run.
- There are 32 Smooth Oracle records: `k=3` and `k=5` for each of the 16 batches. The budget relative error remains at numerical-zero scale.
- The log ends with `Completed; exiting diagnostic run without saving.`

## Interpretation and limits

This confirms that the archived implementation executed the future-gradient diagnostic and passed gradient/finite-difference instrumentation. It does not prove that the Router improves final perceptual quality, that the diagnostic log matches every later source revision, or that the finite-difference proxy is an independent evaluation metric. The full-sampling results and frozen final evaluation remain the authoritative performance evidence.

## Reproduction

```bash
/data/coding/sgmse/env/bin/python experiments/rrsd_future_grad_diagnostic_audit_20261020/audit_log.py
```
