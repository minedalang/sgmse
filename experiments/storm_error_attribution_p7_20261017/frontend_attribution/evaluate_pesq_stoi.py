#!/usr/bin/env python3
"""Run paired PESQ-WB/STOI for the P7-C audio already in this experiment.

Uses the user's mymetrics conda environment. Does not create or change audio.
"""
from __future__ import annotations

import csv
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import soundfile as sf
from pesq import pesq
from pystoi import stoi


PROJECT = Path(__file__).resolve().parents[3]
DATA = PROJECT.parent.parent
EXP = PROJECT / "experiments/storm_error_attribution_p7_20261017"
MANIFEST = EXP / "baseline_manifest.csv"
FRONT = EXP / "frontend_only"
OFF = PROJECT / "experiment_logs/rrsd_tcut_screen/off"
ROUTER = PROJECT / "experiment_logs/rrsd_tcut_screen/router"
OUT = Path(__file__).resolve().parent
SR = 16000
REPS = 10000
SEED = 20261018


def load(path: Path) -> np.ndarray:
    x, sr = sf.read(path, dtype="float32", always_2d=False)
    if x.ndim == 2:
        x = x.mean(axis=-1)
    if sr != SR or x.ndim != 1:
        raise ValueError(f"Invalid audio {path}: sr={sr}, shape={x.shape}")
    if not np.isfinite(x).all():
        raise ValueError(f"Nonfinite values in {path}")
    return np.ascontiguousarray(x)


def ci_mean(x: np.ndarray) -> list[float]:
    rng = np.random.default_rng(SEED)
    ids = rng.integers(0, len(x), size=(REPS, len(x)))
    vals = x[ids].mean(axis=1)
    return [float(np.quantile(vals, 0.025)), float(np.quantile(vals, 0.975))]


def describe(x: np.ndarray) -> dict[str, float]:
    return {"mean": float(x.mean()), "std": float(x.std(ddof=1)),
            "median": float(np.median(x))}


def main() -> None:
    started = time.time()
    with MANIFEST.open(newline="", encoding="utf-8") as f:
        entries = list(csv.DictReader(f))
    if len(entries) != 32:
        raise ValueError(f"Expected 32 entries, got {len(entries)}")
    rows = []
    for i, item in enumerate(entries, 1):
        ref = load(Path(item["clean_path"]))
        arms = {
            "frontend": load(FRONT / item["filename"]),
            "off": load(OFF / item["filename"]),
            "router": load(ROUTER / item["filename"]),
        }
        if any(x.shape != ref.shape for x in arms.values()):
            raise ValueError(f"Length mismatch in {item['filename']}")
        row = {"filename": item["filename"], "uttid": item["uttid"]}
        for arm, enhanced in arms.items():
            row[f"{arm}_pesq_wb"] = float(pesq(SR, ref, enhanced, "wb"))
            row[f"{arm}_stoi"] = float(stoi(ref, enhanced, SR, extended=False))
        rows.append(row)
        if i % 8 == 0:
            print(f"evaluated {i}/{len(entries)}", flush=True)

    summaries = {}
    for metric in ("pesq_wb", "stoi"):
        arrays = {arm: np.array([r[f"{arm}_{metric}"] for r in rows])
                  for arm in ("frontend", "off", "router")}
        comparisons = {}
        for a, b in (("frontend", "off"), ("router", "off"), ("router", "frontend")):
            d = arrays[a] - arrays[b]
            comparisons[f"{a}_minus_{b}"] = {
                "mean_delta": float(d.mean()),
                "median_delta": float(np.median(d)),
                "bootstrap_95ci": ci_mean(d),
                "a_wins": int((d > 0).sum()),
                "b_wins": int((d < 0).sum()),
                "ties": int((d == 0).sum()),
            }
        summaries[metric] = {
            "per_arm": {arm: describe(values) for arm, values in arrays.items()},
            "paired": comparisons,
        }

    with (OUT / "pesq_stoi_per_utterance.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    config = {
        "experiment_id": "storm_error_attribution_p7_20261017_frontend_pesq_stoi",
        "n": len(rows),
        "sample_rate_hz": SR,
        "pesq_mode": "wideband",
        "stoi_extended": False,
        "arms": {"frontend": str(FRONT.relative_to(PROJECT)),
                 "off": str(OFF.relative_to(PROJECT)),
                 "router": str(ROUTER.relative_to(PROJECT))},
        "manifest_sha256": hashlib.sha256(MANIFEST.read_bytes()).hexdigest(),
        "bootstrap": {"replicates": REPS, "seed": SEED,
                      "scope": "exploratory; previously screened development split"},
        "software": {"numpy": np.__version__, "soundfile": sf.__version__,
                     "pesq": "package import", "pystoi": "0.4.1"},
        "training": False,
        "audio_written_or_modified": False,
        "started_unix": started,
        "runtime_seconds": time.time() - started,
        "metrics": summaries,
    }
    (OUT / "pesq_stoi_metrics.json").write_text(
        json.dumps(config, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(summaries, indent=2), flush=True)
    print(f"runtime_seconds={config['runtime_seconds']:.3f}", flush=True)


if __name__ == "__main__":
    main()
