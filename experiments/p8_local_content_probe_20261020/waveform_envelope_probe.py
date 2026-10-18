#!/usr/bin/env python3
"""Separate phase-sensitive waveform changes from short-time envelope changes."""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parent
P8 = ROOT.parent / "p8_frontend_backend_audit_20261020"
WINDOWS = ROOT / "window_metrics.csv"
OUT_CSV = ROOT / "waveform_envelope_windows.csv"
OUT_JSON = ROOT / "waveform_envelope_summary.json"
FS = 16000
F = 320
H = 160
EPS = 1e-12


def center(path):
    x, fs = sf.read(str(path), always_2d=False)
    if fs != FS:
        raise ValueError((path, fs))
    if x.ndim > 1:
        x = x.mean(axis=1)
    return np.asarray(x[9 * FS:21 * FS], dtype=np.float64)


def corr(a, b):
    a = a - np.mean(a)
    b = b - np.mean(b)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + EPS))


def env(x):
    n = 1 + (len(x) - F) // H
    idx = np.arange(F)[None, :] + H * np.arange(n)[:, None]
    z = x[idx] * np.hanning(F)[None, :]
    return np.sqrt(np.mean(z * z, axis=1) + EPS)


def stat(rows, key):
    a = np.asarray([float(r[key]) for r in rows])
    return {"n": int(len(a)), "median": float(np.median(a)), "mean": float(np.mean(a)),
            "q25": float(np.quantile(a, .25)), "q75": float(np.quantile(a, .75))}


def main():
    windows = list(csv.DictReader(WINDOWS.open()))
    manifest = list(csv.DictReader((P8 / "manifest.csv").open()))
    cache = {}
    for i, rec in enumerate(manifest):
        base = Path(rec["noisy_basename"]).name
        paths = {
            "clean": Path(rec["clean_path"]),
            "y_pre": P8 / "outputs" / "y_pre" / f"{i:02d}_{base}",
            "off": P8 / "outputs" / "off" / f"{i:02d}_{base}",
            "router": P8 / "outputs" / "router" / f"{i:02d}_{base}",
        }
        cache[i] = {k: center(p) for k, p in paths.items()}
    out = []
    for r in windows:
        i = int(r["index"])
        a = int(round((float(r["start_s"]) - 9.0) * FS))
        b = int(round((float(r["end_s"]) - 9.0) * FS))
        c = cache[i]["clean"][a:b]
        row = dict(r)
        cenv = env(c)
        for arm in ("y_pre", "off", "router"):
            x = cache[i][arm][a:b]
            row[f"{arm}_wave_corr_clean"] = corr(x, c)
            row[f"{arm}_envelope_corr_clean"] = corr(env(x), cenv)
        row["off_wave_delta_vs_y_pre"] = row["off_wave_corr_clean"] - row["y_pre_wave_corr_clean"]
        row["router_wave_delta_vs_y_pre"] = row["router_wave_corr_clean"] - row["y_pre_wave_corr_clean"]
        row["off_envelope_delta_vs_y_pre"] = row["off_envelope_corr_clean"] - row["y_pre_envelope_corr_clean"]
        row["router_envelope_delta_vs_y_pre"] = row["router_envelope_corr_clean"] - row["y_pre_envelope_corr_clean"]
        out.append(row)
    groups = {
        "all": out,
        "suppression": [r for r in out if r["candidate_frontend_suppression"] == "True"],
        "other": [r for r in out if r["candidate_frontend_suppression"] != "True"],
    }
    summary = {"protocol": {
        "wave_corr": "phase-sensitive gain-invariant waveform Pearson correlation",
        "envelope_corr": "10 ms-hop short-time RMS envelope correlation; content proxy only",
        "warning": "neither metric is a phoneme label or human intelligibility score",
    }, "groups": {}}
    for name, rows in groups.items():
        summary["groups"][name] = {}
        for k in ("y_pre_wave_corr_clean", "off_wave_corr_clean", "router_wave_corr_clean",
                  "y_pre_envelope_corr_clean", "off_envelope_corr_clean", "router_envelope_corr_clean",
                  "off_wave_delta_vs_y_pre", "router_wave_delta_vs_y_pre",
                  "off_envelope_delta_vs_y_pre", "router_envelope_delta_vs_y_pre"):
            summary["groups"][name][k] = stat(rows, k)
    with OUT_CSV.open("w", newline="") as f:
        fields = list(out[0])
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(out)
    OUT_JSON.write_text(json.dumps(summary, indent=2) + "\n")


if __name__ == "__main__":
    main()
