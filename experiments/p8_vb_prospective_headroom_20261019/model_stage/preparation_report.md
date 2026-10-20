# Model preparation: PREPARED ONLY, not a launch permission

Full32conditions/64outputs/104ASR requests and8controls supported. Parent cap64MiB counts only this experiment root, excludes symlink target duplicates. Model files/output rooted model_stage; inputs exclusively input_stage. 170s execution+10s finalization with final180s assertion. No real model/native/metric import smoke, source waveform decoding, enhancement, ASR or training performed by Engineer. All tests use generated temporary signals, fake model/metric modules, or byte hashes. Existing PyTorch/native/Whisper imports remain runtime-only.

Native observer adapted from completed prior attempt02, unchanged parameters/returns, NFE30, true backend norm captured, model weakref released before ASR. Exact offline/local-build/local-probe guard reused unchanged; all sockets refused, unknown refusal checked before next scientific call, including after metric functions. Metrics retain raw finite STOI warning sentinel then null; nonfinite raw becomes null+repr so strict JSON stays valid. Missing metric values never remove a case; paired OFF-front means use both-valid n. Independent checker reimplements candidate choices/error sums/cluster resampling/percentiles/quality bookkeeping and SI-SDR; STOI/PESQ recomputation explicitly reuses same frozen official implementations, not independent algorithm implementations.

Quality official source/API reviewed only: pystoi0.4.1 resamples16kto10k/removes silence, shortframes warn+1e-5; PESQ0.0.4 jointly peak-scales and intrinsic standard processing. No project gain/lag fit or external resampling. No pesq_batch/multiprocessing. Runtime pipeline resampling0 does not mean metrics never resample. Quality input/source mismatch is technical failure; metric exception/null is retained measurement missingness.

Synthetic checks:32FakeTorch-native observer cases,64generated PCM reencodings,225independent DP pairs,104unique ASR requests,96fake-metric generated quality arms,21headroom strata and10000speaker-cluster bootstrap replicate independent checks, SI-SDR degenerate cases, nonfinite strict JSON, paired-null counts, filelike-position/unknown-weight guard, local process grammar and2classified-local/7unknown refusal tests. Exact reports in test JSON. No full real import smoke authorized or performed. Unknown dependencies remain fail-closed, no retries or silent scope reduction.

Worker command AFTER Root review:
```
/data/coding/sgmse/env/bin/python -B MODEL_STAGE/run_pilot.py
```
CPU independent validation AFTER completed run:
```
/data/coding/sgmse/env/bin/python -B MODEL_STAGE/validate_model_run.py
```
Replace MODEL_STAGE with this directory absolute path. Standard config.yaml is byte-identical execution_config.json; runtime writes metrics.json/experiment_report.md here and never overwrites metadata or inputstage records. Original15metadata and failed-source-gate/prior pilot provenance remain locked. No new user feedback claimed. Full scope if runtime budget exceeded becomes preserved technical failure, not downsampling.
