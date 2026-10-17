# Checkpoint data-condition audit (2026-10-18)

## Question

Does the checkpoint configuration show that the Router was trained with the same reverberant condition used in the final evaluation?

## Evidence

The audit loaded `joint_train_logs/sb61i0gj_RRSD/epoch=118-step=80000.ckpt` read-only and recorded its SHA256, epoch, global step, and saved hyperparameters. The saved paths are from the original machine namespace, so they were mapped only for counting and inspection; no checkpoint field was changed.

The checkpoint records:

- `train_wav_scp`: `.../scp/tr_synthetic/wav.scp`
- `valid_wav_scp`: `.../scp/cv_synthetic/wav.scp`
- `train_type=mode2`, `sde=ouve`, `loss_type=score_matching`
- `frontend=storm`, `frontend_trainable=true`
- `rrsd_teacher=future_grad`, `rrsd_freeze_base=true`
- `contrastive_weight=0`, `final_content_weight=0`, `anchor_v2_weight=0`

The mapped SCPs contain 10,800 training entries and 1,200 validation entries. Their noisy paths are the ordinary `.../synthetic_snr-20_-5/noisy/...` paths. The separate final manifest contains matched 150-entry `tt_synthetic_no_reverb` and 150-entry `tt_synthetic_with_reverb` lists. The checkpoint does not record a paired reverb SCP or a reverb augmentation flag.

## Conclusion

The checkpoint metadata is consistent with training on a single non-reverberant synthetic noisy split, while the final evaluation explicitly adds a `with_reverb` condition. This makes training/test condition mismatch a plausible explanation for the matched final-set reverb interaction, but it is not a causal proof: the audit does not isolate frontend conditioning, phase reconstruction, or the Router itself. No new inference or training was run.

## Reproduction and limits

`metrics.json` contains the complete audit values. The original saved SCP paths are not present verbatim in this container; the mapped dataset paths were checked read-only. The dataset and checkpoint were not modified.
