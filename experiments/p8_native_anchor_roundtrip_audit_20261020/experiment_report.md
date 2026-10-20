# Native raw/public anchor audit: failed fixed tolerance, partial7/8

The single authorized run exited1, PID33980/session63955 (completed), after22.149935s. It stopped after case7 when the actual public sampler anchor versus expected waveform roundtrip relative L2 was1.07710186878e−5, exceeding the frozen1e−5 interface tolerance. No tolerance/precision/seed change, additional case, model preflight or retry followed. All partial observations remain retained.

```text
env/bin/python -B experiments/p8_native_anchor_roundtrip_audit_20261020/audit.py
```

Actual counts: source model load1/source EMA materialization1; source PCM reads14; frontend predictions7; backend invocations14; Score calls420 (30 each raw/public branch per case). Planned caps were16PCM/8frontend/16backend/480Score; the eighth selected case was not PCM-read or executed. Its files were only pre/post byte-hashed with the allowed16-source preservation set. No gradients/training/optimizer, ASR, PESQ/STOI, network, install, waveform/spectrum array export or weight copy. Only the source checkpoint was loaded; no N/D or M/P terminal weight was read.

## Fixed interface check and coverage

| dev index / utt | Bin | Raw→actual public anchor relL2 | ExpectedRT→actual anchor relL2 | Noisy-condition relL2 | Interface≤1e−5 | Old raw-MSE sanity |
|---|---|---:|---:|---:|---|---|
| 3 / reader_11121_fileid_10724 | [-20,-18) | 0.245242786351 | 1.64671510321e-06 | 2.2442267348e-06 | True | True |
| 11 / reader_11446_fileid_11151 | [-20,-18) | 0.154110741383 | 5.61830563897e-06 | 4.92106383819e-07 | True | True |
| 18 / reader_11794_fileid_5599 | [-18,-15) | 0.128585224863 | 3.08517255018e-06 | 7.43843706829e-07 | True | True |
| 30 / reader_11128_fileid_5694 | [-18,-15) | 0.183301000289 | 3.45238447063e-06 | 5.54098362646e-07 | True | True |
| 40 / reader_11301_fileid_9495 | [-15,-10) | 0.101081285214 | 3.3298031348e-06 | 4.09099838829e-06 | True | True |
| 43 / reader_11917_fileid_1923 | [-15,-10) | 0.119105524862 | 5.02222413327e-06 | 9.93315998589e-07 | True | True |
| 62 / reader_11053_fileid_1482 | [-10,-5] | 0.126675576837 | 1.07710186878e-05 | 1.38385527766e-07 | False | True |

Case8 (reader_11195_fileid_7449, original dev_index53) is unmeasured. Bins have recorded counts2/2/2/1 in frozen order; no complete8-case aggregate or all-bin success is claimed. The terminal failure must not be recast as completed merely because the seventh scalar discrepancy is small.

All seven actual public normalizers equal the same physical Noisy peak used by the raw branch. Runtime FFT510/hop128/window tensor/exponent0.5/factor0.15/centered256-frame geometry checks passed. Each frontend spectrum was generated once, and public execution used the real model.enhance implementation, observing its actual sampler anchor/condition and ScoreModel.enhance norm. The same original-dev-index seed was reset separately for raw/public branches. Temporary sampler hook and profile were restored in finally; no implementation substitution was made. Raw→public anchor differences are outcomes, not success criteria.

ExpectedRT uses the real frontend waveform synthesis followed by model STFT/pad; actual public anchor includes the physical multiply/divide roundtrip. Non-identical hashes or small expected/actual discrepancies are not silently forced to equality. The1e−5 numerical threshold was set before execution and was not relaxed. Case7 has expected/actual anchor relL2 just above that threshold while its Noisy condition remains within it. This partial failure is not by itself evidence of a normalization bug, historical training mismatch or missing-speech mechanism.

## Saved output-scale observations (partial only)

| dev index | Output raw/public relL2 | Raw→Clean MSE | Public→Clean MSE | Public−Raw MSE |
|---|---:|---:|---:|---:|
| 3 | 1.09018717656 | 0.000203052982236 | 0.000183607909684 | -1.9445072552e-05 |
| 11 | 0.536712444234 | 0.000712294907349 | 0.000420448474467 | -0.000291846432882 |
| 18 | 0.354456803235 | 0.000207268757011 | 0.000197130109699 | -1.01386473128e-05 |
| 30 | 0.621893927423 | 4.01713929673e-05 | 4.30335439203e-05 | +2.86215095295e-06 |
| 40 | 0.0688676535384 | 1.4550317343e-05 | 1.65649852837e-05 | +2.01466794071e-06 |
| 43 | 0.301620340227 | 9.3049303036e-05 | 8.35106798724e-05 | -9.5386231636e-06 |
| 62 | 0.418109459854 | 0.000899596632028 | 0.000889696643519 | -9.8999885092e-06 |

The normalized output difference uses the same Noisy-based scale. Old initial64 raw-MSE sanity passed7/7 within the fixed absolute1e−8/relative1e−4 tolerances; no baseline inference was rerun. These local MSE values do not establish content, perceptual quality or deployment superiority. Substantial raw/public differences in the recorded partial cases only demonstrate sensitivity to this interface path in these observations; this is not a new enhancement method or innovation proof, and no output audio was exported or listened to.

The old CPEL raw-spectrum M/P and N/D results retain their original interface and conclusions. This audit does not rerun or override them. Current source does not establish the interface used in the original300k historical training; unknown history remains explicit. No claim about swallowing causes, true phonemes, hallucination or general model benefit follows.

## Preservation and scalar validation

All42 Worker-protected source/code/config/auth/manifest/receipt/goal/core8/old baseline/M/P/N/D result hashes match before/after, including only the16 selected S/X WAVs and source checkpoint. Script-internal source_pre/source_post also match. No other704-file scan was performed. Existing dirty git state is retained in execution_receipt.json. Bridge39452 and server16022 remain alive and untouched; Root documents, core source, prior experiments and user responses were not modified.

Scalar-only validation reconstructed normalized-MSE and relative-L2 arithmetic for every saved comparison, seven norm equalities,30+30 call counts, interface flags and old-MSE sanity flags. It distinguishes successful bookkeeping from the failed/incomplete scientific audit. It reads saved JSON only and does not reconstruct spectra, decode audio or rerun the model. Shapes/dtypes/tensor hashes remain descriptors in original metrics.json, not exported arrays.

Original metrics SHA256 `c925182dd71d157469cf90c623d0331502412fd6b2d489e6abd5b07ea080c249` remains unchanged. Exact failure traceback, actual call counters, per-case descriptors and script source protection are in metrics.json. Full command/PID/session/timing plus broader42-file protection and resource counts are in execution_receipt.json; scalar checks in independent_validation.json.

The unique audit is stopped with partial evidence. No extra tolerance calibration, inference sweep, model correction, training, seed search or automatic follow-up was started.


Clarification of comparison objects: the large observed anchor difference is Araw versus actual public sampler anchor (recorded relative L2 range 0.101081285214–0.245242786351); the small numerical invariant is expected waveform roundtrip versus actual public anchor, whose seventh case breached1e−5. Output relative L2 ranges 0.0688676535384–1.09018717656 across these7 partial cases. Public Clean MSE is lower in5/7 and higher in2/7, without a complete8-case or general superiority claim. Noisy conditioning also contains numerical roundoff, so output differences cannot all be causally assigned to STFT projection alone. Root limits prior CPEL normal results to their raw-spectrum interface; public deployment equivalence was not established. No old result is rewritten or rerun.
