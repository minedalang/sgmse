# Selective scale 0.25 development screen (2026-10-17)

## Scope

This record completes the reproducibility metadata for an existing 32-utterance development comparison of Router, off, and a selective scale-0.25 policy. It reused the existing paired CSV and summary; no training, inference, dataset modification, or checkpoint overwrite was performed while creating this record.

## Results

| comparison | PESQ | STOI | SI-SDR (dB) |
|---|---:|---:|---:|
| Router - off | +0.019863 [+0.008103, +0.037158] | +0.000967 [+0.000599, +0.001348] | +0.522556 [+0.464653, +0.582461] |
| selective - off | +0.012458 [+0.002103, +0.029679] | +0.000345 [+0.000248, +0.000445] | +0.153090 [+0.136886, +0.170264] |
| selective - Router | -0.007404 [-0.015295, -0.000825] | -0.000622 [-0.000902, -0.000340] | -0.369466 [-0.413488, -0.327086] |

The arm means and full metadata are in `metrics.json`; the paired per-utterance values are in `paired_metrics.csv`.

## Interpretation and limits

On this development split, selective scale 0.25 preserves part of the Router reconstruction gain but is worse than full Router on all three reported metrics. The artifact contains no DNSMOS values, so no DNSMOS claim is made. This split was used for route screening and cannot support final-test claims or further threshold tuning.

## Safety

Original data and valid checkpoints were not modified or overwritten. No cloud push was performed.
