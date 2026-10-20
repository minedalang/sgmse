# 64-speaker CAM++ headroom replication

## Status

Completed inference and CAM++ analysis on a locked speaker-disjoint CV development sample. No training, dataset edits, checkpoint edits, or final-test access occurred.

## Locked sample and reproducibility

- Source: `cv_synthetic/{wav.scp,utt2spk}` under the DNS synthetic tree.
- 158 speakers were present; 141 had at least two utterances. Excluding the 16 speakers in `rrsd_cv_repeat32` and `reader_10658` left exactly 124 eligible speakers.
- Fixed seed `20261017` selected 64 speakers and two deterministic utterances per speaker (128 rows, 30 seconds each).
- Manifest SHA-256: `c04657fa21a33c7f2dc749514cec899767101ba53f46222e9f4ab8577e2f4492`.
- Checkpoint SHA-256: `ab50510ca7fd80798b70f3f0b5365b17448a5f98c75d38755b82212ae2a119c9`.
- CAM++ SHA-256: `07abeeb5150441995b51ea65c9ccc8feed78b33040012f1d2fad29a0e4f5b8d7`.

Off and Router arms used the same 80k checkpoint, per-file seed 2026, PC sampler (`N=30`, `snr=0.5`, no corrector, `mode2`, EMA enabled, RRSD enabled), with `rrsd_scale=0` and `1` respectively. Both arms produced exactly 128 finite, mono, 16 kHz, PCM16 outputs with source hashes unchanged. Inputs were staged using symlinks only.

## CAM++ results

All embeddings use the first 32,000 samples. Pair statistics and intervals use 5,000 speaker-cluster bootstrap replicates (`seed=20261017`); impostor pairs were weighted by speaker-cluster resampling rather than treated as independent.

| arm | paired clean cosine | same-speaker cosine | different-speaker cosine | AUC | EER |
|---|---:|---:|---:|---:|---:|
| noisy | 0.5042 | 0.2525 | 0.1129 | 0.7284 | 0.3281 |
| off | 0.6522 | 0.3828 | 0.1296 | 0.8799 | 0.2049 |
| router | 0.6509 | 0.3852 | 0.1336 | 0.8746 | 0.2188 |
| clean | 1.0000 | 0.6120 | 0.0900 | 0.9663 | 0.0469 |

The predeclared P2b gate passes on this development split: Router versus clean same-speaker deficit is 0.2268 (95% cluster CI [0.1848, 0.2702]); AUC deficit is 0.0916 ([0.0438, 0.1430]); EER excess is 0.1719 ([0.0937, 0.2347]). This is evidence of speaker headroom and does not authorize training or imply final-test generalization.

## Quality guardrails

A complete paired evaluation was subsequently finished on all 128 utterances. Statistics use the same 5,000 speaker-cluster bootstrap replicates and seed as the CAM++ analysis. DNSMOS was requested with `auto`, but the `mymetrics` ONNX Runtime could not initialize its CUDA execution provider (`device_id=0` provider-option parse error), so this run used `CPUExecutionProvider`; this is recorded in `quality_metrics.json` and is not a GPU speed claim.

| metric | off mean | router mean | router − off | 95% cluster CI |
|---|---:|---:|---:|---:|
| PESQ-WB | 1.6201 | 1.6310 | +0.0109 | [+0.0068, +0.0151] |
| STOI | 0.80677 | 0.80792 | +0.00115 | [+0.00092, +0.00137] |
| SI-SDR (dB) | 6.2572 | 6.7687 | +0.5115 | [+0.4818, +0.5426] |
| DNSMOS SIG | 3.4616 | 3.4307 | −0.0309 | [−0.0379, −0.0246] |
| DNSMOS BAK | 4.1415 | 4.1447 | +0.0031 | [−0.0007, +0.0071] |
| DNSMOS OVRL | 3.1645 | 3.1347 | −0.0298 | [−0.0369, −0.0234] |

The router therefore improves reconstruction metrics on this CV sample while reducing DNSMOS SIG/OVRL. It does not provide a quality reason to prefer the Router arm for the proposed Speaker-loss pilot. Earlier terminated attempts and the CUDA provider failure remain recorded in the run log; the final CPU evaluation is the authoritative quality result.

### Offline quality-strata diagnosis

Using the completed per-utterance CSV only, with no new inference, the OVRL decrease is larger at lower estimated input SNR:

| estimated SNR | n | SI-SDR delta | DNSMOS OVRL delta | fraction OVRL<0 |
|---|---:|---:|---:|---:|
| ≤ −15 dB | 45 | +0.6167 dB | −0.0446 | 91.1% |
| (−15, −10] dB | 38 | +0.5184 dB | −0.0318 | 78.9% |
| (−10, −5] dB | 45 | +0.4005 dB | −0.0134 | 75.6% |

Across utterances, SI-SDR delta and OVRL delta have Pearson `r = −0.374` (Spearman `−0.284`); SNR and OVRL delta have Pearson `r = +0.351`. Speaker-mean correlations have the same directions (`r = −0.504` for SI-SDR vs OVRL delta and `r = +0.412` for SNR vs OVRL delta). These are exploratory associations on this CV split, not causal evidence and not a deployable risk predictor. Full values are in `quality_strata_metrics.json`; the reproducible analysis is `analyze_quality_strata.py`.

## Artifacts

- `manifest.csv`, `manifest_lock.json`, `manifest.sha256`
- `config.yaml`, `metrics.json`, `per_utterance_campp.csv`
- `quality_metrics.json`, `quality_per_utterance.csv`
- `output_validation_off_router.json`, `output_manifest_off_router.csv`
- `lock_manifest.py`, `stage_inputs.py`, `validate_outputs.py`, `analyze_campp.py`, `evaluate_quality.py`, `commands.sh`
- Inference logs and outputs under `experiment_logs/speaker_headroom_replication_64_20261017/`

The first failed launches (missing PATH for Ninja, missing `PYTHONPATH` for CAM++, and local analysis key mismatches) are recorded in `off/launch_failure_1.txt` and `campp_failure_note.md`; no repeated inference was performed.
