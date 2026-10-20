# Historical Speaker training parameter audit

## Question

Did the historical masked InfoNCE and Speaker Prototype experiments update the NCSN++ diffusion backend, or did they only update the RRSD Router?

## Evidence checked

- Original training logs: `experiment_logs/rrsd_speaker_pair1000.log`, `rrsd_speaker_pair_masked1000.log`, `rrsd_speaker_proto01_1000.log`, `rrsd_speaker_proto01_cw1_1000.log`, and `rrsd_speaker_svproto_masked_sw05_1000.log`.
- Hyperparameters embedded in the five checkpoints.
- Optimizer state and parameter-group metadata.
- Tensor-by-tensor comparison of each 1,000-step checkpoint against `joint_train_logs/sb61i0gj_RRSD/epoch=118-step=80000.ckpt`.

## Findings

Every arm had `rrsd_freeze_base: true`, `frontend_trainable: true`, one optimizer parameter group at learning rate `3e-5`, and exactly 8 optimizer state entries. The `frontend_trainable` flag is overridden by the freeze path, which sets every parameter to frozen and then re-enables only `rrsd_router.*`.

The original logs report 42.6K trainable parameters and 156M non-trainable parameters. Direct checkpoint comparison confirms this: across all five arms, all 271 `dnn.*` tensors and all 267 `frontend.*` tensors are bitwise unchanged from the 80k EMA base (`diff_norm = 0`); all 8 router tensors changed. The frozen UniSpeech parameters are present in the saved state but have no optimizer state.

## Conclusion

The historical negative Speaker results are valid evidence against the tested **Router-only** adaptation, including masked InfoNCE and prototype variants. They are not evidence that a Speaker–Content loss cannot improve the NCSN++ backend, because backend-only and joint backend/frontend adaptation were not tested.

The next admissible experiment is therefore a low-cost, matched-budget backend-only gradient smoke and then a short paired comparison against a no-speaker-loss backend-only control. It must freeze the StoRM frontend and RRSD router explicitly, use the same 80k EMA initialization, and record trainable parameter names and optimizer groups before any longer run.

No dataset or existing checkpoint was modified.
