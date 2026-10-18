#!/usr/bin/env python3
"""Estimate whether noisy still carries aligned acoustic cues in suppressed windows."""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import soundfile as sf

from probe_local_content import CENTER_START, CENTER_SECONDS, FS, FRAME, HOP, EPS, frames


ROOT = Path(__file__).resolve().parent
P8 = ROOT.parent / "p8_frontend_backend_audit_20261020"
IN_CSV = ROOT / "window_metrics.csv"
OUT_CSV = ROOT / "noisy_cue_windows.csv"
OUT_JSON = ROOT / "noisy_cue_summary.json"


def load_center(path: str | Path) -> np.ndarray:
    x, fs = sf.read(str(path), always_2d=False)
    if fs != FS:
        raise ValueError(f"{path}: expected {FS}, got {fs}")
    if x.ndim > 1:
        x = x.mean(axis=1)
    a = int(CENTER_START * FS)
    return np.asarray(x[a:a + int(CENTER_SECONDS * FS)], dtype=np.float64)


def corr(a: np.ndarray, b: np.ndarray) -> float:
    a = a - np.mean(a)
    b = b - np.mean(b)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + EPS))


def spec_cos(a: np.ndarray, b: np.ndarray) -> float:
    fa = frames(a)
    fb = frames(b)
    la = np.log(np.abs(np.fft.rfft(fa, axis=1)) + EPS)
    lb = np.log(np.abs(np.fft.rfft(fb, axis=1)) + EPS)
    la /= np.linalg.norm(la, axis=1, keepdims=True) + EPS
    lb /= np.linalg.norm(lb, axis=1, keepdims=True) + EPS
    return float(np.median(np.sum(la * lb, axis=1)))


def median(xs):
    xs = np.asarray(xs, dtype=float)
    return float(np.median(xs)) if len(xs) else float("nan")


def stats(rows, key):
    vals = [float(r[key]) for r in rows]
    return {"n": len(vals), "median": median(vals), "mean": float(np.mean(vals)) if vals else float("nan")}


def main() -> None:
    with IN_CSV.open() as f:
        old = list(csv.DictReader(f))
    manifest = list(csv.DictReader((P8 / "manifest.csv").open()))
    out = []
    for rec_i, rec in enumerate(manifest):
        noisy = load_center(rec["noisy_path"])
        clean = load_center(rec["clean_path"])
        for r in [x for x in old if int(x["index"]) == rec_i]:
            a = int(round((float(r["start_s"]) - CENTER_START) * FS))
            b = int(round((float(r["end_s"]) - CENTER_START) * FS))
            nseg, cseg = noisy[a:b], clean[a:b]
            row = dict(r)
            row["noisy_clean_wave_corr"] = corr(nseg, cseg)
            row["noisy_clean_spec_cos_window"] = spec_cos(nseg, cseg)
            row["router_minus_off_rms_db"] = float(r["router_rms_db_vs_clean"]) - float(r["off_rms_db_vs_clean"])
            row["router_minus_off_spec_cos"] = float(r["router_spec_cos_clean"]) - float(r["off_spec_cos_clean"])
            out.append(row)

    suppressed = [r for r in out if r["candidate_frontend_suppression"] == "True"]
    other = [r for r in out if r["candidate_frontend_suppression"] != "True"]
    summary = {
        "protocol": {
            "source": str(P8 / "manifest.csv"),
            "same_windows_as": str(IN_CSV),
            "interpretation": "noisy-clean correlation and spectral cosine are acoustic proxies, not phoneme labels",
        },
        "all_windows": {
            "noisy_clean_wave_corr": stats(out, "noisy_clean_wave_corr"),
            "noisy_clean_spec_cos_window": stats(out, "noisy_clean_spec_cos_window"),
            "router_minus_off_rms_db": stats(out, "router_minus_off_rms_db"),
            "router_minus_off_spec_cos": stats(out, "router_minus_off_spec_cos"),
        },
        "frontend_suppression_windows": {
            "noisy_clean_wave_corr": stats(suppressed, "noisy_clean_wave_corr"),
            "noisy_clean_spec_cos_window": stats(suppressed, "noisy_clean_spec_cos_window"),
            "router_minus_off_rms_db": stats(suppressed, "router_minus_off_rms_db"),
            "router_minus_off_spec_cos": stats(suppressed, "router_minus_off_spec_cos"),
        },
        "non_suppression_windows": {
            "noisy_clean_wave_corr": stats(other, "noisy_clean_wave_corr"),
            "noisy_clean_spec_cos_window": stats(other, "noisy_clean_spec_cos_window"),
            "router_minus_off_rms_db": stats(other, "router_minus_off_rms_db"),
            "router_minus_off_spec_cos": stats(other, "router_minus_off_spec_cos"),
        },
    }
    with OUT_CSV.open("w", newline="") as f:
        fields = list(out[0])
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(out)
    OUT_JSON.write_text(json.dumps(summary, indent=2) + "\n")


if __name__ == "__main__":
    main()
