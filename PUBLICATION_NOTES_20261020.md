# Research/code publication snapshot — 2026-10-20

This commit preserves accumulated source changes, experiment scripts/configurations, saved scalar/tabular evidence, failure records, thesis drafts/readers, and a read-only `research_snapshot/` of the external live research entry. It is NOT evidence of new enhancement results or completed long-run training.

## Included and excluded

- Include the current backend-only training options/helper/unit tests, research tools, experimental source/configuration/JSON/CSV/text logs and reports, thesis sources/generator/figures, and negative results.
- Exclude model/checkpoint files, audio, raw/numerical signal arrays, download/archive bodies, runtime environments, local model caches, numba caches, profiling output and process PID files. Small project-generated plots remain versioned.
- Source paths and SHA records may refer to intentionally unversioned local artifacts. Reproducing those computations requires the documented data/models; Git does not contain their bytes.
- The external live research entry remains `/data/research_plan.md`. `research_snapshot/` is a publication copy with remapped local Markdown navigation, not another execution entry.
- Historical authorization/STARTED records are provenance, not authorization to launch an experiment on another environment.

## Checks actually run

- `env/bin/python -B -m unittest discover -s tests -p test_joint_training_options.py` with offline environment: 3 tests passed. These use a tiny synthetic dataset, not real audio/models.
- AST parsing of 278 pending Python files passed after a minimal historical-validator syntax repair. The original faulty bytes and a repair note are preserved beside that validator. No scientific result was rerun or rewritten by that repair.
- A targeted credential-pattern scan of pending text found no private keys, common API tokens or literal credential matches. This is a heuristic check, not a guarantee against all sensitive information.
- Staged artifact review excludes model/audio/cache extensions and files larger than GitHub's per-file limit. Frozen scientific CSV line endings and archived Markdown hard-break whitespace are preserved rather than reformatted to satisfy generic whitespace checks.

Original valid checkpoints, raw data and past metric values have not been modified. User-authorized removal of two archives is separately documented in the published storage-review receipt. Prepared CPEL 100k data/resume components are not a training execution; the complete long-run driver and real CUDA recovery remain unverified.
