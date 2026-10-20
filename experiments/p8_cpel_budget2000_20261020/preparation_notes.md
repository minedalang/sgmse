# Fixed 2000-step software preparation (not execution)

Only new runner and byte-identical three helpers were created. Old experiments and model sources untouched. No authorization was signed, no main executed, no real model, PCM, checkpoint, CUDA or network accessed.

## Isolation and gates

New fixed protocol restarts each arm from source EMA, 2000 updates; no exact resume, intermediate evaluation, raw readout, or optimizer/raw artifact. Old500 native protocol is validated against old M/P first; explicit exact changed-path allowlist then validates new protocol. SHA locks bind old N/D config, all four old code files, both complete old metrics and public metrics. New auth retains old fields plus `budget_study` equal to complete frozen config dictionary. Baseline identity/provenance retained. Complete original init/eval AST preserved except explicitly normalized exact approved disk-reservation predicate. Runtime `_config_path` and both registry `resolved_path` aliases are individually verified before only those three derived keys are removed for config comparison.

Before each of the first500 optimizer updates, all eight actual draw fields are compared exactly with arm-specific frozen old rows. No loss values or scientific scores select continuation. N requires2000 updates,500 verified prefix records,64 final four-control rows,F/W checks and terminal byte SHA before D. No common_point import/call. Final unchanged raw/precision/normal evaluation retained; actual public main evaluation deliberately separate, needs verified terminal SHA and new authorization later.

## Resources

240MiB directory cap,1GiB free reserve,1200 seconds per arm. Both108MiB terminal slot remaining portions plus unfilled16MiB nonterminal reservation are continuously reserved at every training step, metric write and checkpoint save. Even complete slightly smaller terminal retains its conservative unfilled slot remainder. D alarm also reserves300 public seconds within2700 total using N actual time. Actual text output must stay within16MiB. Failed persistence forcesfailed, records actual resources, writes tiny terminal failure receipt before best-effort full partial persistence. Mandatory failed evidence is not a passed resource gate.

## Actual preparation checks

Four AST parses; actual config/baseline identity/core/source hash/64row and protected init/eval AST validation passed. Synthetic eight-field prefix mismatch rejection passed. Synthetic empty/one/two terminal reservation states, insufficient-free boundary and nonterminal cap rejection passed. Actual gated_config N path ran against memory-only authorization and stopped by deliberate exception at FIRST checkpoint SHA access, with no checkpoint bytes read. Both derived aliases and config path then passed execute-stage validator; incorrect alias rejected. No model imports. Initial metadata test exposed protected-init guard difference and literal counting issue; corrected exact predicate normalization. An initial test import generated only our new pycache; that generated file was removed and all subsequent imports used -B/PYTHONDONTWRITEBYTECODE. No old artifact deleted.

Unverified: actual CUDA graph support, actual prefix equality on real draws, finite training, resources under concurrency,2000 terminal completion and scientific effect. Preparation tests do not authorize execution.

Command after Root review/auth only:
`env/bin/python -B experiments/p8_cpel_budget2000_20261020/run_pilot.py --execute-arm N --config config.yaml`
Then D only after full N gate, same CLI with D. Worker must retain same live handles and no retry.
