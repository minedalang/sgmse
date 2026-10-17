#!/usr/bin/env python3
"""Read-only continuous reverb-severity analysis of frozen final outputs."""
from __future__ import annotations

import csv
import hashlib
import json
import math
import random
from pathlib import Path

import numpy as np
import soundfile as sf
import torch


ROOT = Path(__file__).resolve().parents[2]
EXP = ROOT / "experiments/final_eval_full_20261018"
MANIFEST = ROOT / "experiments/final_eval_manifest_20261018/manifest.csv"
OUT = Path(__file__).resolve().parent
METRICS = ["PESQ_e", "STOI_e", "SI-SDR_e", "DNSMOS_SIG_e", "DNSMOS_OVRL_e", "DNSMOS_P808_e", "DNSMOS_BAK_e"]
BOOT = 10_000
SEED = 20261018
NFFT = 512
HOP = 128


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def load(path: Path) -> np.ndarray:
    x, sr = sf.read(path, dtype="float32", always_2d=False)
    if sr != 16000 or x.ndim != 1 or not np.isfinite(x).all():
        raise ValueError(f"invalid audio {path}: sr={sr}, shape={x.shape}")
    return x.astype(np.float64)


def spec(x: np.ndarray) -> np.ndarray:
    tx = torch.from_numpy(np.asarray(x, dtype=np.float32))
    w = torch.hann_window(NFFT, periodic=True)
    return torch.stft(tx, n_fft=NFFT, hop_length=HOP, win_length=NFFT,
                      window=w, center=True, return_complex=True).numpy()


def severity(no: np.ndarray, rv: np.ndarray) -> dict[str, float]:
    if no.shape != rv.shape:
        raise ValueError(f"length mismatch {no.shape} {rv.shape}")
    den = np.linalg.norm(no) + 1e-12
    sn, sr = spec(no), spec(rv)
    mn, mr = np.abs(sn), np.abs(sr)
    return {
        "clean_wave_rel_l2": float(np.linalg.norm(rv - no) / den),
        "clean_complex_stft_rel_l2": float(np.linalg.norm(sr - sn) / (np.linalg.norm(sn) + 1e-12)),
        "clean_logmag_stft_l1": float(np.mean(np.abs(np.log1p(mr) - np.log1p(mn)))),
        "clean_rms_ratio": float(np.sqrt(np.mean(rv * rv)) / (np.sqrt(np.mean(no * no)) + 1e-12)),
    }


def rankdata(x: np.ndarray) -> np.ndarray:
    order = np.argsort(x, kind="mergesort")
    out = np.empty(len(x), dtype=np.float64)
    i = 0
    while i < len(x):
        j = i + 1
        while j < len(x) and x[order[j]] == x[order[i]]:
            j += 1
        out[order[i:j]] = (i + j - 1) / 2.0 + 1.0
        i = j
    return out


def corr(x: np.ndarray, y: np.ndarray) -> float:
    rx, ry = rankdata(x), rankdata(y)
    rx -= rx.mean(); ry -= ry.mean()
    d = np.sqrt(np.dot(rx, rx) * np.dot(ry, ry))
    return float(np.dot(rx, ry) / d) if d else float("nan")


def bootstrap_corr(x: np.ndarray, y: np.ndarray) -> list[float]:
    rng = random.Random(SEED)
    vals = []
    n = len(x)
    for _ in range(BOOT):
        ii = [rng.randrange(n) for _ in range(n)]
        vals.append(corr(x[ii], y[ii]))
    vals = np.asarray(vals)
    return [float(np.nanquantile(vals, 0.025)), float(np.nanquantile(vals, 0.975))]


def read_scores(arm: str, split: str) -> dict[str, dict[str, float]]:
    path = EXP / "metrics" / arm / f"{split}.csv"
    out = {}
    with path.open() as f:
        for row in csv.DictReader(f):
            fid = Path(row["utt"]).stem.replace("clean_", "")
            out[fid] = {m: float(row[m]) for m in METRICS}
    return out


def main() -> None:
    manifest = {}
    with MANIFEST.open() as f:
        for row in csv.DictReader(f):
            manifest.setdefault(row["file_id"], {})[row["split"]] = row
    ids = sorted(k for k, v in manifest.items() if set(v) == {"no_reverb", "with_reverb"})
    assert len(ids) == 150, len(ids)
    scores = {a: {s: read_scores(a, s) for s in ("no_reverb", "with_reverb")} for a in ("off", "router", "tcut005")}
    rows = []
    for fid in ids:
        no = load(Path(manifest[fid]["no_reverb"]["clean_path"]))
        rv = load(Path(manifest[fid]["with_reverb"]["clean_path"]))
        row = {"file_id": fid, **severity(no, rv)}
        for m in METRICS:
            for arm in ("router", "tcut005"):
                delta_no = scores[arm]["no_reverb"][fid][m] - scores["off"]["no_reverb"][fid][m]
                delta_rv = scores[arm]["with_reverb"][fid][m] - scores["off"]["with_reverb"][fid][m]
                row[f"{arm}_interaction_{m}"] = delta_rv - delta_no
        rows.append(row)
    severity_keys = ["clean_wave_rel_l2", "clean_complex_stft_rel_l2", "clean_logmag_stft_l1", "clean_rms_ratio"]
    out = {
        "experiment_id": "reverb_severity_posthoc_20261018",
        "n": len(rows),
        "unit": "matched_file_id",
        "source_manifest_sha256": sha256(MANIFEST),
        "bootstrap": {"replicates": BOOT, "seed": SEED},
        "correlations": {},
    }
    arr = {k: np.asarray([r[k] for r in rows], dtype=np.float64) for k in severity_keys}
    for sk in severity_keys:
        out["correlations"][sk] = {}
        for arm in ("router", "tcut005"):
            for m in METRICS:
                key = f"{arm}_interaction_{m}"
                y = np.asarray([r[key] for r in rows], dtype=np.float64)
                out["correlations"][sk][key] = {"spearman": corr(arr[sk], y), "ci95": bootstrap_corr(arr[sk], y)}
    with (OUT / "per_file.csv").open("w", newline="") as f:
        fields = list(rows[0])
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(rows)
    (OUT / "metrics.json").write_text(json.dumps(out, indent=2) + "\n")


if __name__ == "__main__":
    main()
