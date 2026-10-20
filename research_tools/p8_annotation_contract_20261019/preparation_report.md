# P8 annotation contract — software preparation only

Status: ready for Research Lead review, **not approved for real data**.

## Implemented

`annotation_contract.py` exports the pure standard-library function specified in
`design.md`. It does no I/O and imports only `decimal` and `re`. Source expected
phones, raw perceived symbols and realized identity remain distinct. Source
addition/deletion/substitution are explicitly scoped to pronunciation, not to
enhancement errors. `err` and terminal `*` remain uncertain rather than silence;
starred raw symbols are preserved. A single `sil` is only a source label, not a
matched absence counterfactual or evidence of exact waveform zero. Manual source
assertions and forced alignment estimates have different provenance fields.
Both approval/eligibility flags are always false. No waveform action, model
abstention, acoustic discernibility or candidate coverage is inferred.

Strict rejections cover bad types, malformed/extra/empty fields, invalid operation,
forced-source manual error triples, contradictory sil operations, identical-phone
substitution, invalid finite nonnegative times and reversed intervals. Decorated
reserved markers (`sil*`, `err*`) have undefined semantics and are rejected.
Symbol validation is ASCII letters/digits and optional terminal star, **not a
verified inventory**; unfamiliar real encodings require review rather than repair.
No TextGrid parser, real dataset reader, CLI, service or global guard was added.

Times are decimal strings without context rounding. Decimal/string inputs retain
their supplied precision; float inputs use their Python decimal string (precision
already lost before the function cannot be restored). Raw textual times and type
names are also saved. Zero duration is allowed and explicitly flagged.

## Actual verification

Run in this directory with Python 3.10.12:

```sh
python3 -B -m unittest -v test_annotation_contract
```

**14 test methods passed, 0 failures/errors/skips**; the initial CLI output reported
0.002 seconds. A second standard-library unittest receipt harness independently
ran the same synthetic tests for machine-readable reporting; measured elapsed
0.0017657110001891851 seconds. See `test_output.txt` and `test_receipt.json`.
Both Python files also passed `ast.parse`. Fixed expected literals are written
independently in the tests; no corpus files are discovered or imported.
These are software checks, not empirical speech metrics or validation of actual
L2-ARCTIC coding, annotator agreement, acoustic truth or method novelty.

## Scope / preservation

First readonly call read the entire attachment: 4551 B, SHA256
`a235e23451a88273189fb9d649904940a868725db05f3dd38983e49290b73a6b`.
Readonly SQLite verified this Engineer thread
`01a152c4-e8b4-7ff3-ac51-f1fd2d0943fc` uses `codex_tools_compat`; bridge PID 39452
was live. AGENTS, the lowercase entry and frozen design were read. Git was already
dirty and was neither reset nor repaired. This directory initially contained only
`design.md`; no earlier implementation was present here. All writes were confined
to this directory, and the design was not edited. No model, audio, annotation
intake, network/download, install, registration, experiment or training ran.

`file_hashes.json` freezes all delivery files except itself (no circular hash).
Root review of this delivery does not approve real corpus use, verifier training,
or modification of StoRM. No empirical `metrics.json` is produced because this is
software preparation, not a research experiment.
