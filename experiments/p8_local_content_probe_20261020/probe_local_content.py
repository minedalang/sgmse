#!/usr/bin/env python3
"""Read-only local content probe for P8.

This deliberately does not claim phoneme recovery.  It measures whether
backend arms add energy in clean-active frames where Y_pre is suppressed,
and whether their local magnitude spectrum moves toward clean.
"""
from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import numpy as np
import soundfile as sf


ROOT = Path(__file__).resolve().parent
P8 = ROOT.parent / "p8_frontend_backend_audit_20261020"
MANIFEST = P8 / "manifest.csv"
OUT_CSV = ROOT / "window_metrics.csv"
OUT_JSON = ROOT / "summary.json"

FS = 16000
CENTER_START = 9.0
CENTER_SECONDS = 12.0
FRAME = 512
HOP = 160
WINDOW = 0.5
EPS = 1e-8


def read(path: str | Path) -> np.ndarray:
    x, fs = sf.read(str(path), always_2d=False)
    if fs != FS:
        raise ValueError(f"{path}: expected {FS} Hz, got {fs}")
    if x.ndim > 1:
        x = x.mean(axis=1)
    lo = int(CENTER_START * FS)
    hi = lo + int(CENTER_SECONDS * FS)
    return np.asarray(x[lo:hi], dtype=np.float64)


def frames(x: np.ndarray) -> np.ndarray:
    n = 1 + max(0, (len(x) - FRAME) // HOP)
    idx = np.arange(FRAME)[None, :] + HOP * np.arange(n)[:, None]
    out = np.zeros((n, FRAME), dtype=np.float64)
    valid = idx < len(x)
    out[valid] = x[np.minimum(idx[valid], len(x) - 1)]
    return out * np.hanning(FRAME)[None, :]


def feat(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    z = frames(x)
    rms = np.sqrt(np.mean(z * z, axis=1) + EPS)
    mag = np.abs(np.fft.rfft(z, axis=1)) + EPS
    logmag = np.log(mag)
    norm = np.linalg.norm(logmag, axis=1) + EPS
    return rms, logmag / norm[:, None]


def median_or_nan(a: np.ndarray) -> float:
    return float(np.median(a)) if len(a) else float("nan")


def main() -> None:
    arms = {"noisy": "noisy", "clean": "clean", "y_pre": "y_pre", "off": "off", "router": "router"}
    rows = []
    summaries = []
    with MANIFEST.open() as f:
        manifest = list(csv.DictReader(f))

    for i, rec in enumerate(manifest):
        paths = {
            "noisy": Path(rec["noisy_path"]),
            "clean": Path(rec["clean_path"]),
            "y_pre": P8 / "outputs" / "y_pre" / f"{i:02d}_{Path(rec['noisy_basename']).name}",
            "off": P8 / "outputs" / "off" / f"{i:02d}_{Path(rec['noisy_basename']).name}",
            "router": P8 / "outputs" / "router" / f"{i:02d}_{Path(rec['noisy_basename']).name}",
        }
        sig = {name: read(path) for name, path in paths.items()}
        fts = {name: feat(x) for name, x in sig.items()}
        clean_rms, clean_spec = fts["clean"]
        # Clean-active is an acoustic activity mask, not a transcript label.
        threshold = max(float(np.quantile(clean_rms, 0.30)), float(np.max(clean_rms) * 0.02))
        active = clean_rms >= threshold
        nwin = int(math.ceil(len(clean_rms) * HOP / FS / WINDOW))
        file_rows = []
        for w in range(nwin):
            a = int(round(w * WINDOW * FS / HOP))
            b = min(len(clean_rms), int(round((w + 1) * WINDOW * FS / HOP)))
            if b <= a:
                continue
            act = active[a:b]
            if not np.any(act):
                continue
            row = {
                "index": i,
                "speaker": rec["speaker"],
                "fileid": rec["fileid"],
                "snr_db": rec["snr_db"],
                "start_s": round(CENTER_START + a * HOP / FS, 3),
                "end_s": round(CENTER_START + b * HOP / FS, 3),
                "active_fraction": float(np.mean(act)),
            }
            # Use only clean-active frames for all content-related summaries.
            for name in arms:
                rms, spec = fts[name]
                rms_local = rms[a:b]
                spec_local = spec[a:b]
                clean_rms_local = clean_rms[a:b]
                clean_spec_local = clean_spec[a:b]
                ratio_db = 20 * np.log10((rms_local[act] + EPS) / (clean_rms_local[act] + EPS))
                # cosine is computed against clean-normalized log spectra.
                cos = np.sum(spec_local[act] * clean_spec_local[act], axis=1)
                row[f"{name}_rms_db_vs_clean"] = median_or_nan(ratio_db)
                row[f"{name}_spec_cos_clean"] = median_or_nan(cos)
            row["off_gain_vs_y_pre_db"] = row["off_rms_db_vs_clean"] - row["y_pre_rms_db_vs_clean"]
            row["router_gain_vs_y_pre_db"] = row["router_rms_db_vs_clean"] - row["y_pre_rms_db_vs_clean"]
            row["off_spec_delta_vs_y_pre"] = row["off_spec_cos_clean"] - row["y_pre_spec_cos_clean"]
            row["router_spec_delta_vs_y_pre"] = row["router_spec_cos_clean"] - row["y_pre_spec_cos_clean"]
            row["candidate_frontend_suppression"] = bool(row["y_pre_rms_db_vs_clean"] <= -8.0)
            row["candidate_backend_energy_fill"] = bool(row["candidate_frontend_suppression"] and row["off_gain_vs_y_pre_db"] >= 2.0)
            row["candidate_router_increment"] = bool(row["candidate_frontend_suppression"] and row["router_gain_vs_y_pre_db"] - row["off_gain_vs_y_pre_db"] >= 1.0)
            file_rows.append(row)
            rows.append(row)

        def vals(key: str, filt=None):
            subset = file_rows if filt is None else [r for r in file_rows if filt(r)]
            return [r[key] for r in subset if np.isfinite(r[key])]

        suppression = [r for r in file_rows if r["candidate_frontend_suppression"]]
        fill = [r for r in file_rows if r["candidate_backend_energy_fill"]]
        summaries.append({
            "index": i,
            "speaker": rec["speaker"],
            "fileid": rec["fileid"],
            "snr_db": int(rec["snr_db"]),
            "n_windows": len(file_rows),
            "n_suppressed_windows": len(suppression),
            "n_energy_fill_windows": len(fill),
            "median_y_pre_rms_db_vs_clean_active": median_or_nan(np.asarray(vals("y_pre_rms_db_vs_clean"))),
            "median_off_rms_db_vs_clean_active": median_or_nan(np.asarray(vals("off_rms_db_vs_clean"))),
            "median_router_rms_db_vs_clean_active": median_or_nan(np.asarray(vals("router_rms_db_vs_clean"))),
            "median_off_gain_vs_y_pre_db": median_or_nan(np.asarray(vals("off_gain_vs_y_pre_db"))),
            "median_router_gain_vs_y_pre_db": median_or_nan(np.asarray(vals("router_gain_vs_y_pre_db"))),
            "median_off_spec_delta_vs_y_pre": median_or_nan(np.asarray(vals("off_spec_delta_vs_y_pre"))),
            "median_router_spec_delta_vs_y_pre": median_or_nan(np.asarray(vals("router_spec_delta_vs_y_pre"))),
        })

    with OUT_CSV.open("w", newline="") as f:
        fields = list(rows[0])
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    OUT_JSON.write_text(json.dumps({"protocol": {
        "source": str(MANIFEST),
        "center_start_s": CENTER_START,
        "center_seconds": CENTER_SECONDS,
        "frame_ms": FRAME / FS * 1000,
        "hop_ms": HOP / FS * 1000,
        "clean_active_quantile": 0.30,
        "suppression_threshold_db": -8.0,
        "energy_fill_threshold_db": 2.0,
        "note": "Acoustic probe only; no phoneme or content recovery claim.",
    }, "files": summaries}, indent=2) + "\n")


if __name__ == "__main__":
    main()
