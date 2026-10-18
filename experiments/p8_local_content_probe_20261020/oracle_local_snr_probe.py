#!/usr/bin/env python3
"""Clean-aware diagnostic only: relate Y_pre suppression to local mixture SNR."""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import soundfile as sf


ROOT = Path(__file__).resolve().parent
P8 = ROOT.parent / "p8_frontend_backend_audit_20261020"
WINDOWS = ROOT / "window_metrics.csv"
OUT_CSV = ROOT / "oracle_snr_windows.csv"
OUT_JSON = ROOT / "oracle_snr_summary.json"
FS = 16000
CENTER_START = 9.0
EPS = 1e-12


def center(path: str | Path) -> np.ndarray:
    x, fs = sf.read(str(path), always_2d=False)
    if fs != FS:
        raise ValueError(f"{path}: expected {FS} Hz, got {fs}")
    if x.ndim > 1:
        x = x.mean(axis=1)
    a = int(CENTER_START * FS)
    return np.asarray(x[a:a + 12 * FS], dtype=np.float64)


def group_stats(rows, key):
    a = np.asarray([float(r[key]) for r in rows], dtype=float)
    return {"n": int(len(a)), "median": float(np.median(a)), "mean": float(np.mean(a)),
            "q25": float(np.quantile(a, .25)), "q75": float(np.quantile(a, .75))}


def main() -> None:
    windows = list(csv.DictReader(WINDOWS.open()))
    manifest = list(csv.DictReader((P8 / "manifest.csv").open()))
    signals = {}
    for i, rec in enumerate(manifest):
        signals[i] = (center(rec["noisy_path"]), center(rec["clean_path"]))
    out = []
    for r in windows:
        i = int(r["index"])
        noisy, clean = signals[i]
        a = int(round((float(r["start_s"]) - CENTER_START) * FS))
        b = int(round((float(r["end_s"]) - CENTER_START) * FS))
        n, c = noisy[a:b], clean[a:b]
        noise = n - c
        row = dict(r)
        row["oracle_local_snr_db"] = float(10 * np.log10((np.mean(c * c) + EPS) / (np.mean(noise * noise) + EPS)))
        row["clean_rms_dbfs"] = float(10 * np.log10(np.mean(c * c) + EPS))
        out.append(row)
    suppressed = [r for r in out if r["candidate_frontend_suppression"] == "True"]
    other = [r for r in out if r["candidate_frontend_suppression"] != "True"]
    summary = {
        "protocol": {
            "purpose": "clean-aware diagnosis, not an inference-time feature",
            "noise_definition": "synthetic noisy waveform minus aligned clean waveform",
            "source_windows": str(WINDOWS),
            "warning": "clean is used only to explain existing suppression windows; no threshold was tuned from this output",
        },
        "all": {"oracle_local_snr_db": group_stats(out, "oracle_local_snr_db"), "clean_rms_dbfs": group_stats(out, "clean_rms_dbfs")},
        "suppression": {"oracle_local_snr_db": group_stats(suppressed, "oracle_local_snr_db"), "clean_rms_dbfs": group_stats(suppressed, "clean_rms_dbfs")},
        "other": {"oracle_local_snr_db": group_stats(other, "oracle_local_snr_db"), "clean_rms_dbfs": group_stats(other, "clean_rms_dbfs")},
    }
    x = np.asarray([float(r["oracle_local_snr_db"]) for r in out])
    y = np.asarray([float(r["y_pre_rms_db_vs_clean"]) for r in out])
    z = np.asarray([float(r["clean_rms_dbfs"]) for r in out])
    summary["correlations"] = {
        "oracle_snr_vs_y_pre_rms_db": float(np.corrcoef(x, y)[0, 1]),
        "clean_rms_vs_y_pre_rms_db": float(np.corrcoef(z, y)[0, 1]),
    }
    with OUT_CSV.open("w", newline="") as f:
        fields = list(out[0])
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(out)
    OUT_JSON.write_text(json.dumps(summary, indent=2) + "\n")


if __name__ == "__main__":
    main()
