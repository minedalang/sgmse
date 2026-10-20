# N/D isolated software preparation — not execution

Only runner plus three byte-identical historical helpers were prepared here. No core model files or old M/P artifacts changed. No checkpoint/WAV bytes were read, no model instantiated, no CUDA, evaluation, optimizer or execution gate invoked. No authorization was created by Engineer.

## Exact contrast

N: native edited-endpoint DSM gradient only. D: that native gradient plus the original dynamic pair-budget component. Common-state Score calls serve pair only. The runner neither imports nor calls common_point; no zero-multiplied pointwise objective is computed. The unchanged helper file still contains its historical common_point definition, unused by this runner. Both arms compute four Score calls and two autograd.grad traversals per update. Native-only first traversal has different graph cost from historical native+common-point M/P; matching call counts does not imply equal actual historical runtime.

Budget dictionary retains historical helper key M_norm, now explicitly interpreted as native-only norm. Both None remain None; pair gradient without native graph support terminates, never fills zeros. N performs the same pair gradient/budget calculations but applies zero. Frozen F/Fourier W, source EMA materialization and terminal EMA remain unchanged. Logged surrogate is not the derivative of a fixed global scalar objective.

## Historical baseline reuse

Both N/D reuse original initial64 without new initial inference. Original baseline identity is retained. The provenance validator checks the old baseline hash and full identity, five historical file hashes, three core-source hashes against saved historical pre-protection, checkpoint identity from metadata only, historical receipt/pre-protection hashes, seed and registry identity. It AST-compares complete evaluation/control/read/transform/frontend/Score/geometry/noise closures and original initialization/freeze/EMA construction against the old runner. Optional current_source_dependencies are current locks only, not invented historical source guarantees. Old sdes/sampler source hashes are not claimed historically verified when absent from old protection.

D gate requires N complete status, 500 updates/rows, terminal EMA hash, current config/source identity, complete final64/four-control evaluation, and F/W unchanged. No automatic resume/retry. Per-arm wall cap 900 s, total declared cap 1800 s; directory cap 320 MiB, free disk reserve 1 GiB. Core network/download/test/ASR restrictions remain unchanged. Only terminal backend EMA state saved.

## Checks performed

- AST parsed four prepared Python files.
- Fixed new config metadata accepted against old scientific configuration.
- Old/new complete initialization/evaluation protected AST equality passed.
- Actual supplied provenance/full old baseline identity and 64 complete rows validated using metadata/source files only.
- Three helper bytes match historical SHA.
- Tiny CPU tensor-only native/pair gradient example: N equals native gradient, D composition matches returned eta, preclip pair budget <= 1%, None/None preserved and native-None/pair-present rejected.
- No model/API/CUDA integration test, no actual graph-support guarantee, no acoustic/scientific result. Prior M/P success does not prove this new gradient collection runs.

## Later execution, only after Root authorization

From project root:

`env/bin/python -B experiments/p8_cpel_native_pair_ablation_20261020/run_pilot.py --execute-arm N --config config.yaml`

Then D only after N meets its completion gate. Root authorization binds config, four code hashes, registries, checkpoint, baseline provenance and core hashes. Engineer preparation is not that authorization.
