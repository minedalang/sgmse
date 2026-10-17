# Final evaluation inference smoke (2026-10-18)

## Purpose

This is a low-cost real-inference chain and cost smoke. It is **not** a quality result and no DNSMOS, PESQ, STOI, SI-SDR, or other quality metric was computed.

## Frozen design

- Manifest: `experiments/final_eval_manifest_20261018/manifest.csv` (SHA256 `4c91782fb1459bd14f2b8581b1f4b537ca98bafd907ecc961226883213bcfa2b`).
- Selected first two rows from each split: 4 utterances total (two `no_reverb`, two `with_reverb`).
- Checkpoint: `joint_train_logs/sb61i0gj_RRSD/epoch=118-step=80000.ckpt`, EMA enabled.
- `joint_enhance_by_dir.py`, PC sampler, `N=30`, corrector `none`, SNR `0.5`, seed `2026`, CUDA device 0.
- Input files were symlinks into the dataset; no dataset files were copied or modified.

## Commands and execution

```bash
PATH="$PWD/env/bin:$PATH" CUDA_VISIBLE_DEVICES=0 env/bin/python -u joint_enhance_by_dir.py --ckpt joint_train_logs/sb61i0gj_RRSD/epoch=118-step=80000.ckpt --in_dir experiments/final_eval_infer_smoke_20261018/input --out_dir experiments/final_eval_infer_smoke_20261018/output/off --use_ema --rrsd --rrsd_scale 0 --N 30 --snr 0.5 --seed 2026 --corrector none
```

`off`: return code 0, wall time 24 s, 4 WAV files, 1,280,176 output bytes.

```bash
PATH="$PWD/env/bin:$PATH" CUDA_VISIBLE_DEVICES=0 env/bin/python -u joint_enhance_by_dir.py --ckpt joint_train_logs/sb61i0gj_RRSD/epoch=118-step=80000.ckpt --in_dir experiments/final_eval_infer_smoke_20261018/input --out_dir experiments/final_eval_infer_smoke_20261018/output/router --use_ema --rrsd --rrsd_scale 1 --N 30 --snr 0.5 --seed 2026 --corrector none
```

`router`: return code 0, wall time 23 s, 4 WAV files, 1,280,176 output bytes.

```bash
PATH="$PWD/env/bin:$PATH" CUDA_VISIBLE_DEVICES=0 env/bin/python -u joint_enhance_by_dir.py --ckpt joint_train_logs/sb61i0gj_RRSD/epoch=118-step=80000.ckpt --in_dir experiments/final_eval_infer_smoke_20261018/input --out_dir experiments/final_eval_infer_smoke_20261018/output/tcut005 --use_ema --rrsd --rrsd_t_cut 0.05 --rrsd_t_cut_scale 0 --N 30 --snr 0.5 --seed 2026 --corrector none
```

`tcut005`: return code 0, wall time 22 s, 4 WAV files, 1,280,176 output bytes.

A preliminary invocation failed before model import because `/usr/bin/time` was absent; it generated no output. The subsequent command used shell timestamps. The first actual import also reported Ninja unavailable despite the package being installed because `env/bin` was not on `PATH`; adding the project environment to `PATH` resolved this. No repeated inference was run after a successful arm.

## Output validation

All three arms produced exactly four readable WAV files. Every file is 16 kHz, 160,000 samples (10.0 s), and 320,044 bytes. The four selected noisy basenames contain their expected `fileid_0`/`fileid_2` identifiers; corresponding clean basenames are `clean_fileid_0.wav` and `clean_fileid_2.wav`. Split-prefixed symlink names keep the two directory trees unambiguous.

The complete per-file and per-arm records are in `metrics.json`; command logs are in `logs/`.

## Interpretation

The frozen checkpoint, routing arms, sampler settings, input mapping, and output writer all execute successfully at this smoke scale. These timings are only a cost/chain check and must not be reported as method quality or full-test performance.
