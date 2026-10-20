# CPEL v2 M/P pilot: terminal configuration-gate failure, no training

The one Root-authorized M command exited1 inside the initial tool call (chunk7f5ba8, tool-reported wall0.000015469s). No running exec session_id or child PID was exposed. The driver failed before STARTED_M.json creation; no PID is invented from an uncreated marker. P was not started because M did not satisfy its completion gate.

```text
env PATH='/data/coding/sgmse/env/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:/command:/usr/local/cuda/bin' env/bin/python -B experiments/p8_cpel_mp_pilot_20261019/run_pilot.py --execute-arm M --config config.yaml
```

Failure chain: gated_config -> validate_v2_config -> validate_fixed_config -> `RuntimeError: Fixed implementation/config mismatch: optimization.learning_rate`.

Static parser diagnosis (not a main rerun): canonical config.yaml is JSON text with learning_rate token `1e-05`. json.loads yields float1e-05; the runner's yaml.safe_load yields string'1e-05'. The frozen numeric equality therefore fails. Config and all four code SHA256 values exactly matched Root authorization before the command. No code/config/parser repair, scientific-condition fallback, parameter tuning or retry was performed.

Actual scientific activity is zero: no source checkpoint model load, PCM decoding, baseline or normal inference, Score call, backward traversal, optimizer step, EMA update, weight/audio export, ASR or network call. No baseline, arm metrics, STARTED_M/P or backend EMA artifact was created. Consequently there are no gradient budget distributions, quality comparisons or content/phoneme conclusions to report. The failure is software gating, not evidence against the research hypothesis.

The authorized raw-byte preservation scan completed before execution:704 unique train/dev WAV files plus22 other selected files (source checkpoint, frozen scripts/config/registries/authorization,3 main sources, selected old metrics, goal attachment and3 user exports). Every one of726 pre/post hashes matches. No extra WAV was decoded for reporting. Compact manifests are training_source_protection_pre.json and training_source_protection_post.json. Existing prior execution_receipt.json was preserved byte-for-byte; the new receipt is training_execution_receipt.json to avoid overwriting that old preflight record.

Canonical config SHA256 is `20f750545ff249238dc2e012d474d0ee1f3137abc8218a93cfea9891894864e5`. Existing dirty git state and exact authorization/code hashes are in training_execution_receipt.json. Bridge39452 and listening server16022 remain alive and untouched. Parent/Root research documents, old artifacts and source materials were not edited. Both arms remain without training results. The sequence is stopped; further correction/execution needs Root instructions.
