# Long-run driver preparation — no real execution

## Files and unchanged science

train_longrun.py is mode/auth/metadata/resource/journal/checkpoint orchestration; training_core.py contains strict source initialization and shared one_update/snapshot/restore. editing.py/loss_helpers.py/gradient_budget.py are byte-identical copies of frozen N/D helpers. resume_io.py remains unchanged. No SGMSE/core source or old experiment edited. Driver imports no ML library before mode authorization; training_core imports SGMSE only inside explicitly invoked source construction.

Full protocol: two source-restarted100k arms,complete10797 registry,batch1,32640sample common crop,4Score/2grad,Adam1e-5,clip1,.999EMA,.01 raw-gradient budget/cap100;common_point absent. Core loader materializes source EMA once then restores original trainable dnn except native FourierW; all other parameters frozen,frontend+dnn eval. No model.train/eval after injection. Global training RNG resets identically after source construction; independent CPU shuffle/crop/parent and actual CUDA device noise/time generators matched. D checks all8 draw fields against ordinary-group complete scientific journal for every retained step; no old256 prefix check or training-score gate.

State snapshots normalize ANY Mapping including actual Torch OrderedDict to plain dict and recursively detached independent CPU tensor clones. They contain raw/EMA/Adam/logcommit/global+local RNG/order/identities. Only actual CUDA device RNG captured/restored; no get_rng_state_all initialization of other GPUs. Restore checks actual live parameter signature and optimizer Parameter-object binding before injection; frozen digests after injection; RNG restored LAST.

## Modes and authorization schema

CLI always requires --mode technical_smoke/full_training/resume_training, --arm both/N/D and explicit absolute local --authorization file. Root status must authorized_once and mode/arm exact. Common auth fields:

- config_sha256
- code_sha256 mapping for train_longrun.py,training_core.py,resume_io.py,editing.py,loss_helpers.py,gradient_budget.py
- core_sha256 exact52 mapping from config
- registry_sha256 exact four registry mapping
- metadata_receipt_sha256, original_cv_metadata_sha256, source_checkpoint_sha256
- analysis_plan_sha256, preparation_scope_sha256

technical_smoke requires arm both, allowed_run_roots exactly [smoke/N/reference,smoke/N/shadow,smoke/D/reference,smoke/D/shadow], and smoke_public_utts exactly:
reader_11103_fileid_4413,reader_11332_fileid_10899,reader_11070_fileid_10064,reader_11076_fileid_2125.

full/resume additionally require technical_smoke_result_path absolute local attempt result and technical_smoke_result_sha256. It must show completed_technical_smoke_only,same complete config/code/core/registry/source/protocol identity,both passed_exact/4retained/6actual and4public calls. Full cannot silently bypass actual technical smoke. D additionally requires ordinary group completed100k sameidentity and all5 valid immutable SHA/content receipts.

resume requires resume:{sha256,completed_step,log_commit}, corresponding existing runs/{arm}/latest_resume.pt. Uncommitted journal tails refused,never truncated/auto-replayed. Every authorization SHA gets exclusive authorizations_used marker, preventing command reuse. New explicit Root authorization required per attempt. Prior attempts' consumed seconds are summed by arm and study; unresolved prior attempt is fail-closed unless auth.prior_attempts_audit maps its absolute attempt path to {path,sha256,attempt_sha256} of Root terminal/elapsed proof with terminal=true,elapsed_seconds>=0. This is explicit external terminal evidence, not PID/timeout guessing. No reset of24h on resume.

## Checkpoints, physical counters and failure handling

Every5k save validated atomic latest; fixed10/25/50/75/100k EMA files exclusively created, full weights-only reload comparison and fsynced immutable receipt. On authorized resume exactly at milestone, missing complete file is created from that restored boundary before next update; existing complete file/content only verified,never overwritten. Existing partial file/receipt fails,requires explicit isolation. Final completed and D admission require all5 SHA/content receipts. No network evaluation during training; future evaluation driver separately authorized (not implemented here).

Scientific train_steps JSONL is append-only/linear. Physical events separately record pid,attempt_id,cumulative physical calls_this_process and logical retained step; resume process counters never mislabelled as whole100k. Smoke keeps reference4step log and checkpoint, copies only own committed2step prefix into distinct shadow,strictly loads shadow latest and replays steps3/4 through same Core.update. It compares full raw/EMA/Adam/RNG/order/logcommit and byte-identical scientific logs. Actual6updates per arm vs retained4 explicitly recorded. No fake checkpoint rewind on reference4step tail.

Smoke N source stage makes4 complete480000sample public calls,one per fixed bin,actual N30/Score30,norm/length/finite/hash/runtime/peak memory and dnn/F/W invariants. No quality score,chunk,crop,clip,ASR or adaptive choice; sampler/profiler wrappers restored in finally. Source load maximum2,one each arm. Smoke must stop; never falls through to full training.

All output/cache/TMPDIR paths confined/data. Guard conservatively reserves complete8GiB future audio+4GiB checkpoint/metadata+600MiB temporary budget even while artifacts accumulate,20GiB directory/10GiB freefloor. Thus it is stricter than merely subtracting remaining artifacts. Real memory/I/O/call counts and underlying CUDA support remain unverified. First source checkpoint byte access is after mode/auth/all metadata/core gates. Source checkpoint/core52/localcode/config/registry/receipt/protocol docs post hashes checked on completion/failure. Best-effort per-attempt terminal failure record retained; no automatic resume/retry or cleanup.

## Actual preparation evidence

- SixAST parses and actual current config/core52/all registry/receipt gate with in-memory authorization passed,zero model/source storage access. Wrong mode and same-identity smoke evidence mismatch rejected; full gate exercised with memory-only passed smoke result.
- CPU synthetic shared Core.update ran4updates/arm: all8draws equal,16Score/8reverse/4optimizer each,N applied0,D budget<=1%. The first toy stub omitted its .dnn property and failed only in CPU test; corrected stub,not scientific code.
- Actual Torch Linear.state_dict OrderedDict normalization to independent plain CPU state passed complete packet validation and weights-only serialization roundtrip. Reviewer-discovered OrderedDict handling was corrected before any real smoke.
- CPU synthetic all5 milestone SHA/content/receipt checks passed; missing current boundary written once,existing valid unchanged,partial immutable refused and retained.
- Cumulative elapsed test deducted30seconds from arm budget; unresolved prior attempt rejected rather than treated aszero.

No source checkpoint,real .pt/WAV,SGMSE instance,CUDA or network accessed; no execution auth generated. CPU tests do not prove real CUDA resume/public30s feasibility or scientific efficacy. Existing earlier CPU exact-resume module toy remains separate evidence. Future evaluator,fullCV1186 complete30s quality/identity pipeline are not implemented by this driver and cannot be claimed complete.

Only after Root review and separate one-shot auth:
`env/bin/python -B experiments/p8_cpel_longrun_20261020/train_longrun.py --mode technical_smoke --arm both --authorization /data/coding/sgmse/experiments/p8_cpel_longrun_20261020/technical_smoke_authorization.json`

This command was NOT executed. No100k authorization or instruction to run is provided.
