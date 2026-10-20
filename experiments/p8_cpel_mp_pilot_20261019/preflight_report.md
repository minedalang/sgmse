# CPEL real-model integration preflight: failed and stopped

The single approved command exited1, PID36540, exec session10351 (completed), with6.194547 seconds measured by the preflight script. Source script: ../preflight_pilot.py. No automatic retry, package installation, implementation/dtype/determinism fallback, or training execution followed.

```text
env/bin/python -B experiments/p8_cpel_mp_pilot_20261019/preflight_pilot.py --execute-preflight
```

Failure: `RuntimeError: Ninja is required to load C++ extensions (pip install ninja to get it)`. Importing sgmse.joint_model -> sgmse.model -> backbone ncsnpp -> upfirdn2d invokes torch.utils.cpp_extension.load; verify_ninja_availability raised before JointSemambaSgmseModel.load_from_checkpoint was reached. The diagnostic's installation suggestion is not an action taken. Full traceback remains in metrics.json / ../preflight_result.json.

Actual counts: checkpoint model loads0 (checkpoint bytes were hashed), PCM/header reads0, native Score0, auxiliary Score0, normal Score0, backward0, optimizer steps0, EMA updates0, parameter updates0, network attempts0, weight/audio exports0. The six explicitly allowed train[0]/dev[0] source WAVs were read as bytes only for pre/post SHA, not PCM decoded. The source checkpoint plus six WAV and four driver source hashes match before/after.

Additivity has no result: neither train nor dev crop was decoded; X=S+N is unknown, not passed or rejected. Gradient norm(pair), norm(M), norm(total) and effective pair-to-M ratio were not reached and remain unknown. Model parameter/buffer digests are unavailable because the model was not constructed; no parameter preservation claim based on an uncomputed digest is made. These are missing measurements, not zero measurements or successful conditions.

All four frozen driver/config bindings match the authorization before/after. Existing dirty git state and exact authorization, source hashes, PID/session/exit status are in ../execution_receipt.json. The compatibility bridge39452 and listening server16022 remain alive and untouched. No Root plan/evidence, main source, source data/checkpoint or human response file was edited.

This subdirectory records the failed preflight only: config.yaml is a byte copy of the dedicated preflight_config.json, metrics.json is a byte copy of preflight_result.json with no recomputation. The parent M/P pilot is not complete and has not begun training. Subsequent integration repair or500-step M/P runs require new Root instructions and authorization. This attempt is stopped.
