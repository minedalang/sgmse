#!/usr/bin/env python3
"""CPU-only grouped CV probe using existing aligned WAVs; performs no inference."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import re
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.stats import spearmanr
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import GroupKFold, LeaveOneGroupOut
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
MANIFEST = ROOT / "experiments/storm_error_attribution_p7_20261017/baseline_manifest.csv"
FRONTEND_DIR = ROOT / "experiments/storm_error_attribution_p7_20261017/frontend_only"
FRONTEND_METRICS = FRONTEND_DIR / "metrics.json"
SAMPLE_RATE = 16000
CHUNK_SAMPLES = SAMPLE_RATE
CLEAN_RMS_MIN = 1e-4
EPS = 1e-12
RIDGE_ALPHA = 1.0
BOOTSTRAP_REPLICATES = 2000
BOOTSTRAP_SEED = 20261017
EXPECTED_MANIFEST_SHA256 = "77b64db9da6a1ff372a22dcda4a6316743133f383103aaffb3dbce2c671c178e"

FEATURE_NAMES = [
    "log_rms_noisy",
    "log_rms_y_pre",
    "rms_ratio_y_pre_over_noisy",
    "rms_difference_y_pre_minus_noisy",
    "band_energy_fraction_noisy_0_1khz",
    "band_energy_fraction_noisy_1_4khz",
    "band_energy_fraction_noisy_4_8khz",
    "band_energy_fraction_y_pre_0_1khz",
    "band_energy_fraction_y_pre_1_4khz",
    "band_energy_fraction_y_pre_4_8khz",
    "spectral_flatness_noisy",
    "spectral_flatness_y_pre",
]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_manifest() -> list[dict[str, str]]:
    observed_manifest_hash = sha256_file(MANIFEST)
    if observed_manifest_hash != EXPECTED_MANIFEST_SHA256:
        raise RuntimeError(
            f"Manifest hash changed: expected {EXPECTED_MANIFEST_SHA256}, "
            f"observed {observed_manifest_hash}"
        )
    with MANIFEST.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 32:
        raise RuntimeError(f"Expected 32 manifest rows, found {len(rows)}")
    if len({row["uttid"] for row in rows}) != len(rows):
        raise RuntimeError("Manifest contains duplicate utterance IDs")
    return rows


def speaker_from_uttid(uttid: str) -> str:
    match = re.fullmatch(r"(reader_\d+)_fileid_\d+", uttid)
    if match is None:
        raise RuntimeError(f"Cannot derive a valid reader ID from uttid={uttid!r}")
    return match.group(1)


def to_mono(audio: np.ndarray) -> np.ndarray:
    if audio.ndim == 1:
        return audio
    if audio.ndim == 2:
        return audio.mean(axis=1)
    raise RuntimeError(f"Expected one- or two-dimensional WAV data, got shape {audio.shape}")


def read_aligned_wav(path: Path, expected_sample_rate: int, expected_samples: int) -> np.ndarray:
    if not path.is_file():
        raise RuntimeError(f"Missing required WAV: {path}")
    audio, sample_rate = sf.read(path, dtype="float64", always_2d=False)
    audio = to_mono(audio)
    if sample_rate != expected_sample_rate:
        raise RuntimeError(f"Expected {expected_sample_rate} Hz at {path}, found {sample_rate}")
    if audio.shape[0] != expected_samples:
        raise RuntimeError(
            f"Expected {expected_samples} mono samples at {path}, found {audio.shape[0]}"
        )
    if not np.isfinite(audio).all():
        raise RuntimeError(f"Non-finite samples in {path}")
    return audio


def expected_frontend_hashes() -> dict[str, str]:
    metrics = json.loads(FRONTEND_METRICS.read_text(encoding="utf-8"))
    files = metrics.get("outputs")
    if not isinstance(files, list):
        raise RuntimeError("Frontend metrics lack per-file hashes")
    hashes = {item["filename"]: item["output_sha256"] for item in files}
    if len(hashes) != 32:
        raise RuntimeError(f"Expected 32 frontend file hashes, found {len(hashes)}")
    return hashes


def waveform_features(noisy: np.ndarray, y_pre: np.ndarray) -> list[float]:
    """Features accept only noisy and Y_pre arrays; no target/output arguments."""
    noisy_rms = float(np.sqrt(np.mean(np.square(noisy))))
    y_pre_rms = float(np.sqrt(np.mean(np.square(y_pre))))
    freqs = np.fft.rfftfreq(CHUNK_SAMPLES, d=1.0 / SAMPLE_RATE)
    one_sided_weights = np.full(freqs.shape, 2.0, dtype=np.float64)
    one_sided_weights[0] = 1.0  # DC is not mirrored.
    one_sided_weights[-1] = 1.0  # 8 kHz Nyquist bin is not mirrored.

    def spectrum_features(signal: np.ndarray) -> tuple[list[float], float]:
        power = np.square(np.abs(np.fft.rfft(signal)))
        energy = power * one_sided_weights
        total_power = float(energy.sum())
        band_masks = [
            (freqs >= 0.0) & (freqs < 1000.0),
            (freqs >= 1000.0) & (freqs < 4000.0),
            (freqs >= 4000.0) & (freqs <= 8000.0),
        ]
        fractions = [float(energy[mask].sum() / max(total_power, 1e-30)) for mask in band_masks]
        power_floor = np.maximum(power, 1e-30)
        flatness = float(np.exp(np.mean(np.log(power_floor))) / np.mean(power_floor))
        return fractions, flatness

    noisy_bands, noisy_flatness = spectrum_features(noisy)
    y_pre_bands, y_pre_flatness = spectrum_features(y_pre)
    return [
        math.log(max(noisy_rms, EPS)),
        math.log(max(y_pre_rms, EPS)),
        y_pre_rms / max(noisy_rms, EPS),
        y_pre_rms - noisy_rms,
        *noisy_bands,
        *y_pre_bands,
        noisy_flatness,
        y_pre_flatness,
    ]


def chunk_target(pre: np.ndarray, router: np.ndarray, clean: np.ndarray) -> tuple[float, float, float]:
    clean_rms = float(np.sqrt(np.mean(np.square(clean))))
    if clean_rms < CLEAN_RMS_MIN:
        raise ValueError("Caller must apply the sole clean-RMS exclusion before target calculation")
    pre_nrmse = float(np.sqrt(np.mean(np.square(pre - clean))) / clean_rms)
    router_nrmse = float(np.sqrt(np.mean(np.square(router - clean))) / clean_rms)
    delta = math.log(max(pre_nrmse, EPS)) - math.log(max(router_nrmse, EPS))
    return delta, pre_nrmse, router_nrmse


def score(y: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    rho = float(spearmanr(y, pred).statistic)
    return {
        "r2": float(r2_score(y, pred)),
        "mae": float(mean_absolute_error(y, pred)),
        "spearman": rho,
    }


def ci(values: list[float]) -> list[float] | None:
    finite = np.asarray([value for value in values if np.isfinite(value)], dtype=np.float64)
    if finite.size == 0:
        return None
    return [float(np.quantile(finite, 0.025)), float(np.quantile(finite, 0.975))]


def grouped_bootstrap(
    y: np.ndarray,
    groups: np.ndarray,
    predictions: dict[str, np.ndarray],
) -> dict[str, object]:
    unique_groups = sorted(set(groups.tolist()))
    indices_by_group: dict[str, np.ndarray] = {
        group: np.flatnonzero(groups == group) for group in unique_groups
    }
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    samples: dict[str, dict[str, list[float]]] = {
        model: {metric: [] for metric in ("r2", "mae", "spearman")}
        for model in predictions
    }
    deltas: dict[str, list[float]] = {metric: [] for metric in ("r2", "mae", "spearman")}
    for _ in range(BOOTSTRAP_REPLICATES):
        drawn = rng.choice(unique_groups, size=len(unique_groups), replace=True)
        indices = np.concatenate([indices_by_group[group] for group in drawn])
        current = {name: score(y[indices], pred[indices]) for name, pred in predictions.items()}
        for model, model_scores in current.items():
            for metric, value in model_scores.items():
                samples[model][metric].append(value)
        for metric in deltas:
            deltas[metric].append(current["ridge"][metric] - current["mean_only"][metric])
    return {
        "method": "speaker_cluster_bootstrap_on_fixed_grouped_oof_predictions",
        "unit": "speaker; all chunks from each sampled speaker travel together",
        "replicates": BOOTSTRAP_REPLICATES,
        "seed": BOOTSTRAP_SEED,
        "confidence_level": 0.95,
        "metric_ci": {
            model: {metric: ci(values) for metric, values in model_samples.items()}
            for model, model_samples in samples.items()
        },
        "ridge_minus_mean_only_ci": {metric: ci(values) for metric, values in deltas.items()},
    }


def main() -> None:
    run_started = time.perf_counter()
    manifest_hash = sha256_file(MANIFEST)
    rows = read_manifest()
    frontend_hashes = expected_frontend_hashes()
    records: list[dict[str, object]] = []
    file_hash_records: list[dict[str, object]] = []
    dropped_tail_samples = 0
    excluded_low_clean_rms = 0
    for row in rows:
        expected_samples = int(row["clean_samples"])
        if not all(
            int(row[key]) == expected_samples
            for key in ("noisy_samples", "router_off_samples")
        ):
            raise RuntimeError(f"Manifest sample-count mismatch for {row['uttid']}")
        if not all(row[key] == "16000" for key in ("noisy_sr", "clean_sr", "router_off_sr")):
            raise RuntimeError(f"Manifest sample-rate mismatch for {row['uttid']}")

        noisy_path = Path(row["noisy_path"])
        clean_path = Path(row["clean_path"])
        router_path = Path(row["router_off_path"])
        y_pre_path = FRONTEND_DIR / row["filename"]
        paths = {
            "noisy": noisy_path,
            "clean": clean_path,
            "y_pre": y_pre_path,
            "router_off": router_path,
        }
        if not (row["all_16khz"] == "True" and row["noisy_off_aligned"] == "True" and row["clean_noisy_aligned"] == "True"):
            raise RuntimeError(f"Manifest alignment flags are not all true for {row['uttid']}")
        if row["filename"] not in frontend_hashes:
            raise RuntimeError(f"Frontend hash missing for {row['filename']}")

        observed_hashes = {key: sha256_file(path) for key, path in paths.items()}
        expected_hashes = {
            "noisy": row["noisy_sha256"],
            "clean": row["clean_sha256"],
            "y_pre": frontend_hashes[row["filename"]],
            "router_off": row["router_off_sha256"],
        }
        for key in paths:
            if observed_hashes[key] != expected_hashes[key]:
                raise RuntimeError(
                    f"SHA-256 mismatch for {key} {row['uttid']}: "
                    f"expected {expected_hashes[key]}, observed {observed_hashes[key]}"
                )

        noisy = read_aligned_wav(noisy_path, SAMPLE_RATE, expected_samples)
        clean = read_aligned_wav(clean_path, SAMPLE_RATE, expected_samples)
        y_pre = read_aligned_wav(y_pre_path, SAMPLE_RATE, expected_samples)
        router = read_aligned_wav(router_path, SAMPLE_RATE, expected_samples)
        n_full_chunks, tail = divmod(expected_samples, CHUNK_SAMPLES)
        dropped_tail_samples += tail
        for chunk_idx in range(n_full_chunks):
            start = chunk_idx * CHUNK_SAMPLES
            stop = start + CHUNK_SAMPLES
            noisy_chunk = noisy[start:stop]
            clean_chunk = clean[start:stop]
            y_pre_chunk = y_pre[start:stop]
            router_chunk = router[start:stop]
            clean_rms = float(np.sqrt(np.mean(np.square(clean_chunk))))
            if clean_rms < CLEAN_RMS_MIN:
                excluded_low_clean_rms += 1
                continue
            target, pre_nrmse, router_nrmse = chunk_target(y_pre_chunk, router_chunk, clean_chunk)
            feature_values = waveform_features(noisy_chunk, y_pre_chunk)
            if not np.isfinite(feature_values).all() or not np.isfinite(target):
                raise RuntimeError(f"Non-finite feature or target for {row['uttid']} chunk {chunk_idx}")
            record: dict[str, object] = {
                "filename": row["filename"],
                "uttid": row["uttid"],
                "speaker_id": speaker_from_uttid(row["uttid"]),
                "chunk_index": chunk_idx,
                "chunk_start_sample": start,
                "clean_rms": clean_rms,
                "nrmse_y_pre": pre_nrmse,
                "nrmse_router_off": router_nrmse,
                "delta_log_waveform_nrmse": target,
            }
            record.update(dict(zip(FEATURE_NAMES, feature_values)))
            records.append(record)

        file_hash_records.append(
            {
                "filename": row["filename"],
                "uttid": row["uttid"],
                "paths": {key: str(path) for key, path in paths.items()},
                "sha256": observed_hashes,
                "sample_rate_hz": SAMPLE_RATE,
                "samples": expected_samples,
            }
        )

    if not records:
        raise RuntimeError("No eligible one-second chunks remain")
    groups = np.asarray([record["speaker_id"] for record in records], dtype=object)
    y = np.asarray([record["delta_log_waveform_nrmse"] for record in records], dtype=np.float64)
    X = np.asarray([[record[name] for name in FEATURE_NAMES] for record in records], dtype=np.float64)
    speakers = sorted(set(groups.tolist()))
    speaker_counts = Counter(groups.tolist())

    if len(speakers) >= 5:
        splitter = GroupKFold(n_splits=5)
        cv_name = "per_speaker_grouped_5_fold"
    else:
        splitter = LeaveOneGroupOut()
        cv_name = "leave_one_speaker_out_fallback_fewer_than_5_groups"
    mean_predictions = np.full(y.shape, np.nan, dtype=np.float64)
    ridge_predictions = np.full(y.shape, np.nan, dtype=np.float64)
    fold_for_chunk = np.full(y.shape, -1, dtype=np.int64)
    fold_records = []
    for fold_id, (train_idx, test_idx) in enumerate(splitter.split(X, y, groups=groups), start=1):
        mean_predictions[test_idx] = float(np.mean(y[train_idx]))
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X[train_idx])
        X_test = scaler.transform(X[test_idx])
        model = Ridge(alpha=RIDGE_ALPHA)
        model.fit(X_train, y[train_idx])
        ridge_predictions[test_idx] = model.predict(X_test)
        fold_for_chunk[test_idx] = fold_id
        fold_records.append(
            {
                "fold": fold_id,
                "train_chunks": int(len(train_idx)),
                "test_chunks": int(len(test_idx)),
                "train_speakers": int(len(set(groups[train_idx].tolist()))),
                "test_speakers": int(len(set(groups[test_idx].tolist()))),
                "test_speaker_ids": sorted(set(groups[test_idx].tolist())),
            }
        )
    if np.isnan(mean_predictions).any() or np.isnan(ridge_predictions).any() or (fold_for_chunk < 1).any():
        raise RuntimeError("CV did not generate exactly one out-of-fold prediction per chunk")

    predictions = {"mean_only": mean_predictions, "ridge": ridge_predictions}
    pooled_scores = {model_name: score(y, pred) for model_name, pred in predictions.items()}
    comparison = {
        "ridge_minus_mean_only": {
            "r2": pooled_scores["ridge"]["r2"] - pooled_scores["mean_only"]["r2"],
            "mae": pooled_scores["ridge"]["mae"] - pooled_scores["mean_only"]["mae"],
            "spearman": pooled_scores["ridge"]["spearman"] - pooled_scores["mean_only"]["spearman"],
        }
    }
    bootstrap = grouped_bootstrap(y, groups, predictions)
    counts_per_utterance = Counter(record["uttid"] for record in records)
    improved_count = int(np.count_nonzero(y > 0.0))
    worsened_count = int(np.count_nonzero(y < 0.0))
    tie_count = int(np.count_nonzero(y == 0.0))
    metrics = {
        "experiment_id": "p7c_observable_repair_probe_20261017",
        "status": "exploratory_not_held_out_or_confirmatory",
        "question": "Can deployment-observable noisy/Y_pre features predict local Router-off repair vs harm relative to Y_pre?",
        "execution": {
            "device": "cpu",
            "gpu_used": False,
            "training_run": False,
            "inference_run": False,
            "dataset_or_checkpoint_modified": False,
            "analysis_runtime_seconds": None,
        },
        "input_validation": {
            "manifest_path": str(MANIFEST),
            "manifest_sha256": manifest_hash,
            "expected_manifest_sha256": EXPECTED_MANIFEST_SHA256,
            "manifest_rows": len(rows),
            "unique_uttids": len({row['uttid'] for row in rows}),
            "frontend_manifest_entries": len(frontend_hashes),
            "all_required_wavs_present_and_hash_verified": True,
            "all_files_16khz_and_mono_sample_lengths_aligned": True,
            "utterance_input_hashes": file_hash_records,
        },
        "specification": {
            "chunk_duration_seconds": 1.0,
            "chunk_samples": CHUNK_SAMPLES,
            "full_chunks_before_exclusion": sum(int(row["clean_samples"]) // CHUNK_SAMPLES for row in rows),
            "dropped_incomplete_tail_samples": dropped_tail_samples,
            "excluded_chunks_clean_rms_below_1e-4": excluded_low_clean_rms,
            "eligible_chunks": len(records),
            "target": "log(NRMSE(Y_pre, clean)) - log(NRMSE(Router-off, clean)); natural log; positive means Router-off helps",
            "nrmse_definition": "sqrt(mean((estimate-clean)^2)) / sqrt(mean(clean^2))",
            "target_clean_usage": "offline target and sole eligibility threshold only",
            "feature_names": FEATURE_NAMES,
            "feature_count": len(FEATURE_NAMES),
            "feature_sources": ["noisy input", "Frontend-only Y_pre"],
            "router_output_used_as_predictive_input": False,
            "feature_selection_after_target_inspection": False,
            "spectral_definition": "rectangular 1-s FFT power; fractional energy in [0,1k), [1k,4k), [4k,8k]; spectral flatness geometric/mean power with 1e-30 floor",
        },
        "grouping": {
            "speaker_id_rule": "uttid prefix before _fileid_ (validated reader_<digits>_fileid_<digits>)",
            "utterances": len(counts_per_utterance),
            "speakers": len(speakers),
            "speaker_ids": speakers,
            "chunks_per_speaker": {speaker: int(speaker_counts[speaker]) for speaker in speakers},
            "cv": cv_name,
            "folds": fold_records,
        },
        "out_of_fold_metrics": pooled_scores,
        "model_comparison": comparison,
        "speaker_cluster_bootstrap": bootstrap,
        "target_distribution": {
            "n_chunks": int(len(y)),
            "mean": float(np.mean(y)),
            "sample_standard_deviation": float(np.std(y, ddof=1)),
            "median": float(np.median(y)),
            "p05": float(np.quantile(y, 0.05)),
            "p95": float(np.quantile(y, 0.95)),
            "minimum": float(np.min(y)),
            "maximum": float(np.max(y)),
        },
        "descriptive_backend_help_fraction": {
            "definition": "fraction of eligible one-second chunks with delta_log_waveform_nrmse > 0",
            "help_chunks": improved_count,
            "worsen_chunks": worsened_count,
            "tie_chunks": tie_count,
            "eligible_chunks": int(len(y)),
            "fraction_help": float(improved_count / len(y)),
            "interpretation": "descriptive only; chunks are clustered and not independent confirmation",
        },
        "limitations": [
            "Repeatedly screened 32-item P7-C development split; exploratory only, not held-out or confirmatory.",
            "Chunks within utterances and speakers are correlated; segment count does not equal independent sample count.",
            "The 30-speaker grouped bootstrap is approximate and operates on fixed grouped out-of-fold predictions.",
            "Local waveform NRMSE is a reconstruction proxy, not a perceptual metric; no DNSMOS or local perceptual claims are made.",
        ],
    }

    csv_path = OUT / "per_chunk_predictions.csv"
    csv_fields = [
        "filename",
        "uttid",
        "speaker_id",
        "chunk_index",
        "chunk_start_sample",
        "fold",
        "clean_rms",
        "nrmse_y_pre",
        "nrmse_router_off",
        "delta_log_waveform_nrmse",
        *FEATURE_NAMES,
        "pred_mean_only",
        "pred_ridge",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=csv_fields)
        writer.writeheader()
        for index, record in enumerate(records):
            output = dict(record)
            output.update(
                {
                    "fold": int(fold_for_chunk[index]),
                    "pred_mean_only": float(mean_predictions[index]),
                    "pred_ridge": float(ridge_predictions[index]),
                }
            )
            writer.writerow(output)
    metrics["outputs"] = {
        "per_chunk_predictions_csv": str(csv_path),
        "per_chunk_predictions_csv_sha256": sha256_file(csv_path),
        "analysis_script": str(Path(__file__).resolve()),
        "analysis_script_sha256": sha256_file(Path(__file__)),
    }
    metrics["execution"]["analysis_runtime_seconds"] = float(time.perf_counter() - run_started)
    (OUT / "metrics.json").write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"out_of_fold_metrics": pooled_scores, "grouping": metrics["grouping"], "descriptive_backend_help_fraction": metrics["descriptive_backend_help_fraction"]}, indent=2))


if __name__ == "__main__":
    main()
