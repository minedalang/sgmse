# CPEL preflight attempt02: failed and stopped

This was one separately Root-authorized attempt after the preserved attempt01. Actual command exited1, PID37385, completed exec session93436, script wall time10.518409s. No retry or scientific condition changes followed.

```text
env PATH='/data/coding/sgmse/env/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:/command:/usr/local/cuda/bin' /data/coding/sgmse/env/bin/python -B /data/coding/sgmse/experiments/p8_cpel_mp_pilot_20261019/preflight_attempt02/preflight_pilot.py --execute-preflight
```

Exact child PATH was bound in authorization and is retained in execution_receipt.json. `ninja` resolved to `/data/coding/sgmse/env/bin/ninja`, version `1.13.2.git.kitware.jobserver-pipe-1`. No package was installed. Model import reached load_from_checkpoint, so the earlier Ninja failure did not recur. There was no CLI extension/cache stdout or stderr; existing cache reuse versus rebuild was not separately instrumented and is not claimed. No source/ops substitution was made.

New failure: `TypeError("JointWaveDataModule.__init__() missing 1 required positional argument: 'base_dir'")`. During the single load_from_checkpoint invocation, the JointSemambaSgmseModel constructor called ScoreModel.__init__, which called data_module_cls without required base_dir (sgmse/model.py159). The checkpoint read/load attempt therefore did not complete model construction, strict state loading or EMA materialization. Complete traceback remains in metrics.json and ../preflight_result.json. No constructor/API correction was made by Worker.

Actual calls: native Score0, auxiliary Score0, normal Score0, backward0, optimizer0, EMA updates0. PCM/header reads0, network attempts0, weight/audio exports0. Only the six permitted source WAVs were hashed as bytes by the authorized script; they were not decoded. Source checkpoint, six WAVs and four driver files match pre/post SHA256. The four scientific scripts, config, preflight config and registry copies were byte-identical to the reviewed parent versions before execution. No source writes occurred.

Additivity is unknown because crops were not read. Pair/M/total gradient norms and effective ratio are unknown because no Score graph was constructed. Parameter/buffer digest was not reached; preservation is not inferred from an uncomputed digest. Neither these missing measurements nor the runtime failure establish a scientific failure or success of CPEL.

All seven preserved attempt01 artifacts remain byte-identical. This attempt's complete command/PATH/Ninja version, actual PID/session/exit, config/code/source hashes and failure are in ../execution_receipt.json. Compatibility bridge39452 and listening server16022 remain alive and untouched. No Root document or main source was edited. The three files in this subdirectory record attempt02 only: config is a dedicated-config byte copy; metrics is the original failed-result byte copy, with no reanalysis.

Attempt02 is stopped. Parent M/P training remains unexecuted and incomplete; no root_execution_authorization.json exists. Future correction/execution requires new Root instructions.
