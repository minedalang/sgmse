# One-file 10720 router-off provenance replay

Completed successfully. The single actual model replay exactly reproduced the saved full 30 s off WAV, byte for byte and sample for sample. Frontend Y_pre also exactly matched its saved full 30 s WAV after normal PCM16 serialization. No router replay was needed or performed.

The explicitly authorized purpose was to test causal provenance of already completed output, rather than repeat a research experiment. Original config.yaml, original off_retry.log, original report, engineer_findings.md and source code were inspected before launch. Original logs record RRSD disabled, EMA enabled, N30 and seed2026; config/report specify PC, corrector none, SNR0.5 and CUDA0. The literal historical shell command and historical script hashes are unavailable. The command below is reconstructed from those known settings. Current repository has preexisting modifications; git state and current code hashes are recorded in attempt02/metrics.json, so historical code equality is not claimed.

## Locked artifacts

- Original noisy source: `/data/datasets/dns_ins20/dns_snr_-20_-5/synthetic_snr-20_-5/noisy/book_04733_chp_0031_reader_10929_13_WIPwL0T99nw-10Ebq7HBVxk-U_5z8BeHx5M_snr-16_fileid_10720.wav`.
- Input SHA256: `cb714d176ad32c2aa041dc5a71715349ec9c35ae441bf55b0036f37d5bec173c`.
- Input symlink basename preserved: `04_book_04733_chp_0031_reader_10929_13_WIPwL0T99nw-10Ebq7HBVxk-U_5z8BeHx5M_snr-16_fileid_10720.wav`.
- Seed formula: `(2026 + little_endian_uint32(BLAKE2b(basename, digest_size=4))) mod 2^32`; effective seed **2997368683**.
- Checkpoint: `/data/coding/sgmse/joint_train_logs/sb61i0gj_RRSD/epoch=118-step=80000.ckpt`.
- Checkpoint SHA256: `ab50510ca7fd80798b70f3f0b5365b17448a5f98c75d38755b82212ae2a119c9`.
- Saved original output: `/data/coding/sgmse/experiments/p8_frontend_backend_audit_20261020/outputs/off/04_book_04733_chp_0031_reader_10929_13_WIPwL0T99nw-10Ebq7HBVxk-U_5z8BeHx5M_snr-16_fileid_10720.wav`.
- New replay output: `/data/coding/sgmse/experiments/p8_10720_pairing_forensics_20261018/replay_off/attempt02/output/04_book_04733_chp_0031_reader_10929_13_WIPwL0T99nw-10Ebq7HBVxk-U_5z8BeHx5M_snr-16_fileid_10720.wav`.
- Identical saved/replay SHA256: `912f380d90be61b6b2ad0f8bb4cab166aaf597c5e0423eae26193aa4207d1b03`.
- Loaded train_type mode2, frontend storm, residual refinement false, RRSD false. Checkpoint hyperparameters contain no separate frontend checkpoint reference; see checkpoint_reference_details.json.

## Result

| Comparison | Samples | Max absolute difference | Difference RMS | Correlation |
|---|---:|---:|---:|---:|
| Full off output | 480000 | 0 | 0 | 1 |
| 13.75–15.75 s off window | 32000 | 0 | 0 | 1 |
| Full frontend after PCM16 serialization | 480000 | 0 | 0 | 1 |

Frontend pre/post length match and noisy lengths are all 480000. No length crop/pad occurred. Saved/replay off RMS is 0.0027741598691380211; window RMS is 0.0027878582917004492. Frontend float32 array hash and serialized hash are recorded; the comparison accounts for standard output PCM16 quantization. No new listening crops or normalization variants were saved.

Actual GPU: cuda:0, NVIDIA L40S, UUID GPU-f9aa822f-67b6-c2aa-c2f9-eeabe9ab7cde. Torch 2.14.1+cu130, runtime CUDA 13.0. Actual sampler NFE **30**. Enhance wall time 9.163 s; replay process work wall time 19.396 s. The wrapper captured existing enhance(timeit=True) timing/NFE and returned its waveform unchanged; it also observed the existing frontend length matching function without changing its returned tensor. No model source file edits were made.

## Command and execution record

Existing inference entry point arguments:

```sh
joint_enhance_by_dir.py --ckpt /data/coding/sgmse/joint_train_logs/sb61i0gj_RRSD/epoch=118-step=80000.ckpt --in_dir /data/coding/sgmse/experiments/p8_10720_pairing_forensics_20261018/replay_off/attempt02/inputs --out_dir /data/coding/sgmse/experiments/p8_10720_pairing_forensics_20261018/replay_off/attempt02/output --device cuda:0 --sample_rate 16000 --train_type mode2 --sampler_type pc --N 30 --corrector none --snr 0.5 --seed 2026 --use_ema
```

Executed via `attempt02/run_replay.py`, with existing repository env/bin and CUDA/bin prepended to PATH. The wrapper imports the same entry point, sets the arguments above and calls main once. No random calls were added by its observation probes.

Initial launch `run_replay.py` failed during imports before model/checkpoint load because ninja was not on PATH. It is already installed at env/bin/ninja. The failure metrics.json and run.log were preserved. The corrected launch in attempt02 added existing paths and completed one inference; no downloads or package installation occurred. All launch and result artifacts were created in new directories. Input, saved output and checkpoint SHA256 were rechecked after replay and remained unchanged.

This exact replay strongly supports causal provenance from the intended noisy file to the saved off output under the recovered settings. It does not establish word/phoneme identity or classify hallucination. Historical process state was not recorded; the exact match closes the practical artifact provenance question without claiming access to that historical state.

Successful full metrics: attempt02/metrics.json. Initial failed launch metrics: metrics.json. New WAV is 960044 bytes; all audit files together remain below 20 MB. No checkpoint or original audio was overwritten.
