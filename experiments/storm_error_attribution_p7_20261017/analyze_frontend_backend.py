#!/usr/bin/env python3
"""Clean-aware P7-C diagnostics for paired frontend and StoRM outputs.

This script describes this already-screened 32-utterance development split. It
does not choose or tune an inference policy and does not claim deployable
reliability prediction.
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
import statistics
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch


ROOT = Path(__file__).resolve().parents[2]
EXP = ROOT / "experiments/storm_error_attribution_p7_20261017"
MANIFEST = EXP / "baseline_manifest.csv"
FRONT_DIR = EXP / "frontend_only"
OFF_DIR = ROOT / "experiment_logs/rrsd_tcut_screen/off"
ROUTER_DIR = ROOT / "experiment_logs/rrsd_tcut_screen/router"
OUT = EXP / "frontend_attribution"
SR = 16000
N_FFT = 510
HOP = 128
BOOTSTRAP_REPS = 10000
BOOTSTRAP_SEED = 20261017


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_mono(path: Path) -> tuple[np.ndarray, int]:
    x, sr = sf.read(path, dtype="float32", always_2d=False)
    if x.ndim == 2:
        x = x.mean(axis=-1)
    if x.ndim != 1:
        raise ValueError(f"Expected mono audio after downmix, got {x.shape}: {path}")
    if not np.isfinite(x).all():
        raise ValueError(f"Non-finite waveform: {path}")
    return x, int(sr)


def si_sdr(reference: np.ndarray, estimate: np.ndarray) -> float:
    reference = reference.astype(np.float64)
    estimate = estimate.astype(np.float64)
    reference = reference - reference.mean()
    estimate = estimate - estimate.mean()
    denom = float(np.dot(reference, reference)) + 1e-12
    target = (float(np.dot(estimate, reference)) / denom) * reference
    noise = estimate - target
    return 10.0 * math.log10((float(np.dot(target, target)) + 1e-12) /
                             (float(np.dot(noise, noise)) + 1e-12))


def stft(x: np.ndarray) -> np.ndarray:
    tensor = torch.from_numpy(np.ascontiguousarray(x, dtype=np.float32))
    window = torch.hann_window(N_FFT, periodic=True)
    return torch.stft(
        tensor,
        n_fft=N_FFT,
        hop_length=HOP,
        win_length=N_FFT,
        window=window,
        center=True,
        return_complex=True,
    ).numpy()


def spectral_metrics(clean: np.ndarray, estimate: np.ndarray) -> dict[str, float]:
    c = stft(clean)
    e = stft(estimate)
    if c.shape != e.shape:
        raise ValueError(f"STFT mismatch: clean={c.shape}, estimate={e.shape}")
    cmag = np.abs(c)
    emag = np.abs(e)
    denom = float(cmag.sum()) + 1e-12
    mag_error = np.abs(emag - cmag)
    complex_error = np.abs(e - c)
    # Phase is reported only on bins with appreciable clean energy. This is a
    # clean-aware diagnostic mask, not a deployable inference feature.
    phase_mask = cmag >= max(float(np.quantile(cmag, 0.995)) * 1e-2, 1e-10)
    phase_weight = cmag * phase_mask
    phase_cos = np.cos(np.angle(e) - np.angle(c))
    phase_error = float((phase_weight * (1.0 - phase_cos)).sum() /
                        (phase_weight.sum() + 1e-12))

    freqs = np.fft.rfftfreq(N_FFT, d=1.0 / SR)[: cmag.shape[0]]
    bands = [("low_0_1k", 0.0, 1000.0), ("mid_1_4k", 1000.0, 4000.0),
             ("high_4_8k", 4000.0, 8000.1)]
    result = {
        "mag_l1_rel": float(mag_error.sum() / denom),
        "complex_l1_rel": float(complex_error.sum() / denom),
        "phase_cos_error_clean_active": phase_error,
        "phase_active_fraction": float(phase_mask.mean()),
    }
    for name, low, high in bands:
        mask = (freqs >= low) & (freqs < high)
        result[f"{name}_mag_l1_rel"] = float(mag_error[mask].sum() /
                                               (cmag[mask].sum() + 1e-12))
        result[f"{name}_complex_l1_rel"] = float(complex_error[mask].sum() /
                                                    (cmag[mask].sum() + 1e-12))
    return result


def bootstrap_mean_ci(values: np.ndarray) -> tuple[float, float]:
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    n = len(values)
    indices = rng.integers(0, n, size=(BOOTSTRAP_REPS, n))
    means = values[indices].mean(axis=1)
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def summarize(rows: list[dict[str, object]], metric: str, arm: str) -> dict[str, float]:
    values = np.asarray([float(r[f"{arm}_{metric}"]) for r in rows], dtype=np.float64)
    return {
        "mean": float(values.mean()),
        "std": float(values.std(ddof=1)),
        "median": float(np.median(values)),
    }


def paired_summary(rows: list[dict[str, object]], metric: str,
                   better: str = "higher") -> dict[str, object]:
    # Difference is frontend-only minus Router-off; positive always favors
    # frontend for higher-is-better metrics, and negative for lower-is-better.
    f = np.asarray([float(r[f"frontend_{metric}"]) for r in rows])
    o = np.asarray([float(r[f"off_{metric}"]) for r in rows])
    delta = f - o
    if better == "lower":
        wins = int((delta < 0).sum())
        oriented = -delta
    else:
        wins = int((delta > 0).sum())
        oriented = delta
    lo, hi = bootstrap_mean_ci(oriented)
    return {
        "delta_frontend_minus_off_mean": float(delta.mean()),
        "delta_frontend_minus_off_median": float(np.median(delta)),
        "bootstrap_95ci_oriented_toward_frontend": [lo, hi],
        "frontend_better_utterances": wins,
        "off_better_utterances": int(len(rows) - wins - np.sum(delta == 0)),
        "ties": int(np.sum(delta == 0)),
    }


def main() -> None:
    if OUT.exists():
        raise FileExistsError(f"Refusing to overwrite existing result directory: {OUT}")
    started = time.time()
    OUT.mkdir(parents=True)
    with MANIFEST.open(newline="", encoding="utf-8") as f:
        manifest = list(csv.DictReader(f))
    if len(manifest) != 32:
        raise ValueError(f"Expected the existing 32-file manifest, got {len(manifest)}")

    rows: list[dict[str, object]] = []
    audio_manifest = []
    for entry in manifest:
        name = entry["filename"]
        paths = {
            "clean": Path(entry["clean_path"]),
            "noisy": Path(entry["noisy_path"]),
            "frontend": FRONT_DIR / name,
            "off": OFF_DIR / name,
            "router": ROUTER_DIR / name,
        }
        waves = {}
        for key, path in paths.items():
            if not path.is_file():
                raise FileNotFoundError(path)
            wave, sr = load_mono(path)
            if sr != SR:
                raise ValueError(f"Expected {SR} Hz, got {sr}: {path}")
            waves[key] = wave
        lengths = {k: int(v.size) for k, v in waves.items()}
        if len(set(lengths.values())) != 1:
            raise ValueError(f"Waveform length mismatch for {name}: {lengths}")
        if lengths["noisy"] != int(entry["noisy_samples"]):
            raise ValueError(f"Input length differs from baseline manifest for {name}")

        row: dict[str, object] = {
            "filename": name,
            "uttid": entry["uttid"],
            "length": lengths["clean"],
            "noisy_peak": float(np.max(np.abs(waves["noisy"]))),
            "clean_rms": float(np.sqrt(np.mean(waves["clean"] ** 2))),
        }
        for arm in ("frontend", "off", "router"):
            estimate = waves[arm]
            row[f"{arm}_rms"] = float(np.sqrt(np.mean(estimate ** 2)))
            row[f"{arm}_peak"] = float(np.max(np.abs(estimate)))
            row[f"{arm}_si_sdr"] = si_sdr(waves["clean"], estimate)
            row[f"{arm}_waveform_nrmse"] = float(
                np.sqrt(np.mean((estimate - waves["clean"]) ** 2)) /
                (np.sqrt(np.mean(waves["clean"] ** 2)) + 1e-12)
            )
            row.update({f"{arm}_{k}": v for k, v in
                        spectral_metrics(waves["clean"], estimate).items()})

        rows.append(row)
        audio_manifest.append({
            "filename": name,
            "uttid": entry["uttid"],
            "length": lengths["clean"],
            **{f"{key}_sha256": sha256(path) for key, path in paths.items()},
        })

    metrics_to_report = [
        ("si_sdr", "higher"),
        ("waveform_nrmse", "lower"),
        ("mag_l1_rel", "lower"),
        ("complex_l1_rel", "lower"),
        ("phase_cos_error_clean_active", "lower"),
        ("low_0_1k_mag_l1_rel", "lower"),
        ("mid_1_4k_mag_l1_rel", "lower"),
        ("high_4_8k_mag_l1_rel", "lower"),
    ]
    summary: dict[str, object] = {
        "n": len(rows),
        "arms": {},
        "paired_frontend_vs_off": {},
        "paired_router_vs_off": {},
    }
    for metric, better in metrics_to_report:
        summary["arms"][metric] = {arm: summarize(rows, metric, arm)
                                   for arm in ("frontend", "off", "router")}
        summary["paired_frontend_vs_off"][metric] = paired_summary(rows, metric, better)
        # The arm field naming is frontend/off; compare router as the first arm
        # through a temporary projected copy to keep the same paired helper.
        # paired_summary uses metric-prefixed fields; project Router into its
        # frontend slot without changing the source rows or audio files.
        projected = [dict(r, **{
            f"frontend_{metric}_paired": r[f"router_{metric}"],
            f"off_{metric}_paired": r[f"off_{metric}"],
        }) for r in rows]
        summary["paired_router_vs_off"][metric] = {
            **paired_summary(projected, f"{metric}_paired", better),
            "delta_router_minus_off_mean": float(np.mean([
                float(r[f"router_{metric}"]) - float(r[f"off_{metric}"]) for r in rows
            ])),
        }

    csv_path = OUT / "per_utterance_metrics.csv"
    fieldnames = list(rows[0].keys())
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    (OUT / "paired_audio_manifest.json").write_text(
        json.dumps(audio_manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    (OUT / "metrics.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    config = {
        "experiment_id": "storm_error_attribution_p7_20261017_frontend_attribution",
        "purpose": "Clean-aware diagnostic comparison of frontend-only, Router-off and RRSD Router outputs",
        "source_manifest": str(MANIFEST.relative_to(ROOT)),
        "source_manifest_sha256": sha256(MANIFEST),
        "n": len(rows),
        "split_warning": "Previously used development split; not an independent blind test.",
        "arms": {
            "frontend": str(FRONT_DIR.relative_to(ROOT)),
            "off": str(OFF_DIR.relative_to(ROOT)),
            "router": str(ROUTER_DIR.relative_to(ROOT)),
        },
        "stft": {"sample_rate_hz": SR, "n_fft": N_FFT, "hop_length": HOP,
                 "window": "periodic Hann", "center": True},
        "bands_hz": [[0, 1000], [1000, 4000], [4000, 8000]],
        "phase_mask": "per utterance clean magnitude >= 1% of its 99.5th percentile; clean-aware diagnostic only",
        "bootstrap": {"replicates": BOOTSTRAP_REPS, "seed": BOOTSTRAP_SEED,
                      "interpretation": "exploratory paired interval; not a confirmatory test"},
        "metrics": [name for name, _ in metrics_to_report],
        "quality_evaluators": "PESQ/STOI/DNSMOS not run in this diagnostic stage",
        "training": False,
        "dataset_modified": False,
        "checkpoint_modified": False,
        "runtime_seconds": time.time() - started,
        "software": {"python_torch": torch.__version__, "numpy": np.__version__,
                     "soundfile": sf.__version__},
    }
    (OUT / "config.yaml").write_text(json.dumps(config, indent=2, ensure_ascii=False) + "\n",
                                       encoding="utf-8")

    print(json.dumps({"out": str(OUT), "n": len(rows),
                      "paired_frontend_vs_off": summary["paired_frontend_vs_off"],
                      "runtime_seconds": config["runtime_seconds"]}, indent=2))


if __name__ == "__main__":
    main()
