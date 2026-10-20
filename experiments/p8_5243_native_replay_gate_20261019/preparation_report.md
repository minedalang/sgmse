Prepared only; no model imports, checkpoint loads, inference or training.

Entrypoint is run_replay.py with no options. Supervisor refuses existing start/result/output artifacts, freezes prepared file hashes, launches one child, enforces 180 seconds and 15,000,000 bytes, and records failures without retry. Current source code, source WAV/clip and inherited checkpoint hashes are locked; the checkpoint is read/hash-checked before model import and once after replay within the same deadline. No model/weight/data/core/helper files were edited.

The unchanged joint_enhance_by_dir.main receives a new-directory symlink bearing the original 00_ basename and frozen original mode2/80k EMA/PC N30/OFF/base2026 settings, effective seed369124093. Full30s frontend context and original normalization are retained. Exactly one frontend and one backend are required. Runtime hooks capture raw matched frontend and raw enhance return .npy before clipping/serialization, normalization peak, NFE/settings, CPU/all CUDA RNG states around frontend/backend/sampler, and first trace state shapes/dtypes/hashes. Trace callback occurs AFTER predictor; x_t is pre-predictor after corrector (none corrector is identity), x_next and x_mean are post-predictor, and callback RNG is post-predictor. No trace state is blindly labeled initialxT. No trace state arrays are saved. RNG arrays are retained for subsequent matched replay.

Historical raw5243 frontend is unavailable. Historical checks are PCM/byte gates only, not historical raw equality. Frontend raw is re-encoded in memory using PCM16 and compared byte-for-byte and sample-for-sample before allowing backend. The original main clips backend output to [-1,1] and serializes WAV; its full30s bytes/PCM and [300000:332000] crop bytes/PCM must match historical OFF/clip. Raw backend remains separately available before clipping. No interventions, Router, classifiers, evaluation, sweeps or training.

Static checks: AST parse, five core hashes match both successful10720 and CPU audit, source/clip hashes and effective seed, NumPy scalar JSON serialization and tiny synthetic in-memory PCM serialization. No actual replay was run. metrics.json and experiment_report.md are intentionally reserved for Worker.

Worker command after Research Lead review:

`/data/coding/sgmse/env/bin/python /data/coding/sgmse/experiments/p8_5243_native_replay_gate_20261019/run_replay.py`

Do not change config or driver once Worker starts. Any mismatch/failure stops this gate with no retry. Exact historical literal invocation/code hashes are unavailable; runtime invocation is a supported reconstruction.

Pre-run correction after Research Lead review: match hook now captures the original noisy tensor peak before PCM decoding and uses separate frontend_pcm/historical_frontend_pcm names. An actual-hook AST CPU mock (no torch/model import) verified distinct noisy/frontend peaks 0.75/0.25, unchanged returned matched frontend, exact PCM/bytes gate, JSON progress roundtrip and rejection of corrupt historical bytes. Replay remains prepared only pending explicit Research Lead review and Worker launch.
