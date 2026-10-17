# Official final evaluation: Router, off, and t-cut 0.05 (2026-10-18)

## Frozen protocol

This is the first formal evaluation on the locked official DNS Ins20 synthetic test manifest. The manifest has 150 `no_reverb` and 150 `with_reverb` utterances (SHA256 `4c91782fb1459bd14f2b8581b1f4b537ca98bafd907ecc961226883213bcfa2b`). The checkpoint and routing settings were frozen before this run; no tuning was performed.

- Checkpoint: `joint_train_logs/sb61i0gj_RRSD/epoch=118-step=80000.ckpt`
- EMA enabled; PC sampler, `N=30`, no corrector, SNR `0.5`, seed `2026`
- GPU inference: `CUDA_VISIBLE_DEVICES=0`
- Three arms: `off` (`rrsd_scale=0`), `router` (`rrsd_scale=1`), `tcut005` (`rrsd_t_cut=0.05`, `rrsd_t_cut_scale=0`)
- Each split had an independent symlink input tree and output tree. The dataset was not copied or modified.

## Inference cost and outputs

| arm | no_reverb | with_reverb | total |
|---|---:|---:|---:|
| off | 479 s | 479 s | 958 s |
| router | 478 s | 478 s | 956 s |
| tcut005 | 478 s | 479 s | 957 s |

All six inference directories produced 150 readable WAV files. Every output is 16 kHz and 160000 samples (10 seconds). Per-file SHA256 and byte totals are in `metrics/*_outputs.json`; command logs are in `logs/`.

## Metric evaluation

Each split and arm was evaluated separately with:

```bash
/data/coding/mymetrics/env/bin/python /data/coding/mymetrics/get_metrics_for_papertable_forDNS.py \
  --clean_dir ... --noisy_dir ... --enhanced_dir ... \
  --dnsmos --dnsmos_device cuda --save_csv ...
```

The requested DNSMOS CUDA provider was unavailable: ONNX Runtime emitted `No registered plugin EP device found for 'CUDAExecutionProvider'`. The evaluator verified this and fell back to CPU. This is a runtime warning only; all six metric runs completed and wrote 150-row CSVs. DNSMOS was therefore CPU-evaluated, while inference was GPU-evaluated.

### Enhanced corpus means

| split | arm | PESQ | STOI | SI-SDR | DNSMOS SIG | DNSMOS OVRL | DNSMOS P808 | DNSMOS BAK |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| no_reverb | off | 3.0350 | 0.9752 | 18.989 | 3.635 | 3.395 | 4.096 | 4.154 |
| no_reverb | router | 3.0614 | 0.9757 | 19.308 | 3.630 | 3.394 | 4.098 | 4.161 |
| no_reverb | tcut005 | 3.0568 | 0.9757 | 19.263 | 3.631 | 3.394 | 4.098 | 4.160 |
| with_reverb | off | 1.3200 | 0.3596 | -12.785 | 2.990 | 2.549 | 3.348 | 3.907 |
| with_reverb | router | 1.3053 | 0.3575 | -12.621 | 2.965 | 2.520 | 3.325 | 3.924 |
| with_reverb | tcut005 | 1.3085 | 0.3571 | -12.733 | 2.970 | 2.527 | 3.336 | 3.923 |

### Paired differences relative to off

Values are mean per-utterance differences (arm minus off), with 95% percentile bootstrap CIs. The bootstrap unit is the utterance, seed `20261018`, 10,000 replicates. Full values are in `metrics/paired_bootstrap.pretty.json`.

| split | arm | PESQ | STOI | SI-SDR | DNSMOS SIG | DNSMOS OVRL | DNSMOS P808 | DNSMOS BAK |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| no_reverb | router | +0.026383 [0.023147, 0.029634] | +0.000522 [0.000393, 0.000661] | +0.319033 [0.302189, 0.335560] | -0.004805 [-0.007129, -0.002520] | -0.000204 [-0.003241, 0.003024] | +0.002224 [-0.001156, 0.005458] | +0.007423 [0.004208, 0.011058] |
| no_reverb | tcut005 | +0.021817 [0.018910, 0.024684] | +0.000459 [0.000343, 0.000584] | +0.273827 [0.258430, 0.289350] | -0.003909 [-0.006018, -0.001874] | -0.000099 [-0.002888, 0.002748] | +0.001675 [-0.001291, 0.004408] | +0.006091 [0.003174, 0.009409] |
| with_reverb | router | -0.014714 [-0.036809, 0.003592] | -0.002149 [-0.003142, -0.001167] | +0.163791 [0.085537, 0.244340] | -0.024959 [-0.036654, -0.012990] | -0.028741 [-0.042369, -0.015840] | -0.023010 [-0.036303, -0.009909] | +0.016659 [0.004038, 0.029468] |
| with_reverb | tcut005 | -0.011483 [-0.039525, 0.013971] | -0.002492 [-0.003310, -0.001694] | +0.051472 [-0.019601, 0.117180] | -0.019399 [-0.030156, -0.008233] | -0.022446 [-0.035153, -0.010211] | -0.011710 [-0.023691, 0.000230] | +0.015780 [0.003904, 0.027678] |

For completeness, pooled 300-utterance paired deltas are in `metrics/paired_bootstrap.pretty.json`. Pooled Router versus off: PESQ `+0.005834 [-0.005825, +0.015530]`, STOI `-0.000813 [-0.001344, -0.000309]`, SI-SDR `+0.241412 [+0.202065, +0.282450]`, DNSMOS SIG `-0.014882 [-0.021161, -0.008582]`, OVRL `-0.014472 [-0.021656, -0.007403]`, P808 `-0.010393 [-0.017405, -0.003574]`, BAK `+0.012041 [+0.005612, +0.018725]`. Pooled t-cut versus off: PESQ `+0.005167 [-0.009441, +0.018450]`, STOI `-0.001016 [-0.001468, -0.000575]`, SI-SDR `+0.162650 [+0.124887, +0.198189]`, DNSMOS SIG `-0.011654 [-0.017242, -0.006063]`, OVRL `-0.011272 [-0.017827, -0.005035]`, P808 `-0.005018 [-0.011248, +0.001160]`, BAK `+0.010936 [+0.004766, +0.017360]`.

## Interpretation

This locked final evaluation shows a split-dependent effect. On `no_reverb`, Router improves PESQ, STOI, and SI-SDR with small DNSMOS SIG reduction. On `with_reverb`, Router improves SI-SDR but lowers PESQ, STOI, and DNSMOS SIG/OVRL/P808. The t-cut arm does not remove this split-dependent tradeoff. These are the first frozen final-set measurements and should not be used to retune thresholds.

## Warnings, failures, and storage

- A previous smoke experiment is separate and untouched.
- No checkpoint or dataset file was changed; no cloud push was performed.
- The only metric warning was DNSMOS CUDA provider unavailability and automatic CPU fallback. No metric run failed; all CSVs contain 150 rows.
- `/data` had approximately 4.5 GB free before inference and approximately 4.2 GB after outputs and logs. The experiment uses split symlinks; generated WAVs and CSVs are the material additions.
- Total GPU inference wall time was 2,871 seconds (47.85 minutes) across six arms/splits. DNSMOS metric evaluation ran on CPU and took roughly 15 minutes in total.
