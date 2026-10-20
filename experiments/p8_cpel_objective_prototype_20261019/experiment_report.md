# CPEL stage0 synthetic objective prototype: passed

Unique recorded synthetic run: 18 engineering checks passed in 2.017077s, PID25589, exec session2565 (completed). Unique independent scalar checker: 11 saved fixed arithmetic cases / 126 scalar values passed in 0.032390s, PID25773; completed in initial tool call (no running session_id). Maximum scalar absolute error: 1.7763568394002505e-15. Both exit0; no retry or failures. The18 engineering checks and11 saved arithmetic cases are distinct counts.

Scope is synthetic CPU float64/complex128 tensor arithmetic and autograd, including real/complex finite-difference gradcheck. No model class, CUDA API, real audio/header, source dataset, checkpoint, network, training, optimizer step or fitting was used. CUDA_VISIBLE_DEVICES was empty; OMP_NUM_THREADS=1 and MKL_NUM_THREADS=1. Commands were launched directly as fixed argv without shell recursion or command recovery.

Commands (each once, cwd=this experiment directory):

```text
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 ../../env/bin/python -B synthetic_check.py --output metrics.json
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 ../../env/bin/python -B reference_check.py metrics.json --output independent_validation.json
```

Engineering checks passed:

- aligned finite zero
- reverse penalized
- overgain upper
- positive margin zero response penalized
- zero margin permits zero
- cosine tiny perfect yet amplitude penalized
- orthogonal energy
- soft mask nonidempotent outside
- complex Hermitian projection
- zero tiny skipped finite zero gradient
- valid only batch mean
- common offset invisible
- Noisy passthrough noise target skipped not rejected
- common state distinct kernel targets read exact clean
- native shared epsilon states differ
- Score grad with caller detached state anchor
- tiny autograd finite difference gradcheck
- complex gradcheck

Independent reference_check imports neither objective.py nor torch and reconstructs the saved fixed-case fields plus OUVE scalar identities with Python complex/scalar arithmetic. It does not independently verify gradients, acoustic validity, real-model behavior or enhancement benefit. The OUVE checks establish kernel arithmetic identities only, not common-proposal DSM correctness or posterior recoverability. Caller-detached state/anchor retain a finite gradient for a synthetic trainable Score scalar; this is not a real Score-network integration test.

Known limitations are exhibited, not repaired: positive lower margin imposes amplitude; zero lower margin admits zero response; cosine can be perfect for tiny output; common errors cancel in differences; speech Noisy-passthrough can satisfy the directional objective while also passing noise. In the noise-only fixture clean delta=0, the clean direction is undefined and the entire sample (including outside penalty) is skipped. Its separate input-direction gain=1 and response energy=1 show nonzero passed noise; zero objective is not noise suppression success. Zero edit with identical input/RNG is consistency, not learned acoustic evidence. Margins/weights/threshold are software examples and not approved training hyperparameters.

Preservation: six frozen code/config/notes/analysis hashes match before and after; ten selected files remain identical, including sgmse/model.py, joint_model.py, joint_data_module.py, formal final metrics, prior statistical/probe metrics, current goal attachment and three user-chat exports (including completed94 and partial12). No WAV hashes were computed. Existing dirty git state is retained in execution_receipt.json. Provider codex_tools_compat / bridge39452 and server16022 remained alive and untouched.

Metrics SHA256: `30b088cb2f1fcc47c3c5dcd9e0ac1f3abb6e7e7162c91b6403021b2188efaba3`. Independent validation SHA256: `c753d9f58346a2d2e8d0ceb971603dd407a591cbac329c3fb23b89fc47ae428b`. Full commands, timings, actual PID/handle, git state and pre/post preservation hashes are recorded in execution_receipt.json.

Stage0 is complete and stopped. No acoustic success, identifiability, hallucination suppression or novelty claim follows. Model integration and M/P training remain separately gated by Root review; Root plan/evidence were not edited.
