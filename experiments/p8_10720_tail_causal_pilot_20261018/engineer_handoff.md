# Worker handoff

Prepared only; no model inference or experiment execution performed by engineer.

Execute once from repository root:

```bash
PATH=/data/coding/sgmse/env/bin:$PATH /data/coding/sgmse/env/bin/python experiments/p8_10720_tail_causal_pilot_20261018/run_pilot.py > experiments/p8_10720_tail_causal_pilot_20261018/run.log 2>&1
```

Six prescribed backend-off runs execute with one loaded checkpoint and one recomputed float frontend. Stop gates: unchanged source/code/checkpoint hashes; frontend exact float32 hash; no_op exact saved PCM16 WAV byte hash; input changes strictly absent outside intervention window; global normalization peak unchanged; NFE30 each run. Frontend recomputation precedes all masks. Seeds reset per enhance; CPU/CUDA RNG states are hashed. No core model edits.

Outputs: raw NumPy arrays and PCM16 WAV per run; original float frontend, intervention mask, edited anchor/condition arrays; provenance; metrics; report; raw fixed20ms frame CSV. Speech activity uses full16kHz clean reference, disjoint20ms frames, fixed−70dBFS RMS threshold. Tail energy is Parseval mean-square4–7.5kHz with explicit unwindowed FFT. Outside/active waveform relative L2 compares every raw output against new no_op raw. PCM16 metrics are labeled reference only. No output metric uses the old quantized backend as its raw comparator.

Synthetic CPU checks and Python parsing passed; relevant current model/driver hashes match the successful SNR control. Validation is in engineer_validation.json. No CUDA or model load was performed for these checks.

Execution refuses existing/partial artifacts. On failure it writes metrics/report and failure_archive/failure_snapshot.json, preserves partial outputs in place, exits nonzero, and must not be retried automatically. Worker should inspect the failure and report to Research Lead.

Oracle clean tail is an OOD mechanism probe, not deployment. Prior negative P8-M on5243 active masking is a distinct experiment. Seed repetitions measure stochastic risk and are not ensemble selection or a novelty claim.
