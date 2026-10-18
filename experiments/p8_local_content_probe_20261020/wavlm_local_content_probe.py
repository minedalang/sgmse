#!/usr/bin/env python3
"""Frozen WavLM local content proxy on all Y_pre-suppression windows.

This is diagnostic only. It uses clean to quantify an upper-bound reference and
does not train or create an inference-time feature.
"""
from __future__ import annotations

import csv
import json
import argparse
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from transformers import WavLMModel

ROOT = Path(__file__).resolve().parent
P8 = ROOT.parent / "p8_frontend_backend_audit_20261020"
WINDOWS = ROOT / "window_metrics.csv"
OUT_CSV = ROOT / "wavlm_local_windows.csv"
OUT_JSON = ROOT / "wavlm_local_summary.json"
MODEL = "/root/.cache/huggingface/hub/models--microsoft--wavlm-base-plus/snapshots/4c66d4806a428f2e922ccfa1a962776e232d487b"
FS = 16000
MAX_SAMPLES = 8000
BATCH = 32
EPS = 1e-8


def center(path):
    x, fs = sf.read(str(path), always_2d=False)
    if fs != FS:
        raise ValueError((path, fs))
    if x.ndim > 1:
        x = x.mean(axis=1)
    return np.asarray(x[9 * FS:21 * FS], dtype=np.float32)


def norm_pad(x):
    x = x - x.mean()
    x = x / (np.max(np.abs(x)) + 1e-6)
    if len(x) < MAX_SAMPLES:
        x = np.pad(x, (0, MAX_SAMPLES - len(x)))
    return x[:MAX_SAMPLES]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--group", choices=("suppression", "all"), default="suppression")
    args = parser.parse_args()
    windows = list(csv.DictReader(WINDOWS.open()))
    if args.group == "suppression":
        windows = [r for r in windows if r["candidate_frontend_suppression"] == "True"]
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

    items = []
    for wi, r in enumerate(windows):
        i = int(r["index"])
        a = int(round((float(r["start_s"]) - 9.0) * FS))
        b = int(round((float(r["end_s"]) - 9.0) * FS))
        for arm in ("clean", "y_pre", "off", "router"):
            items.append((wi, arm, norm_pad(cache[i][arm][a:b])))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = WavLMModel.from_pretrained(MODEL, local_files_only=True).eval().to(device)
    feats = {layer: {} for layer in (3, 6, 12)}
    with torch.inference_mode():
        for start in range(0, len(items), BATCH):
            batch = items[start:start + BATCH]
            x = torch.from_numpy(np.stack([z[2] for z in batch])).to(device)
            hs = model(x, output_hidden_states=True, return_dict=True).hidden_states
            for layer in (3, 6, 12):
                h = hs[layer].float()
                h = h / (h.norm(dim=-1, keepdim=True) + EPS)
                for j, (wi, arm, _) in enumerate(batch):
                    feats[layer][(wi, arm)] = h[j].cpu().numpy()
            del x, hs
            if device.type == "cuda":
                torch.cuda.empty_cache()

    out = []
    for wi, r in enumerate(windows):
        for layer in (3, 6, 12):
            ref = feats[layer][(wi, "clean")]
            for arm in ("y_pre", "off", "router"):
                h = feats[layer][(wi, arm)]
                out.append({
                    "window_index": wi,
                    "source_index": int(r["index"]),
                    "fileid": r["fileid"],
                    "start_s": r["start_s"],
                    "end_s": r["end_s"],
                    "layer": layer,
                    "arm": arm,
                    "cosine_to_clean": float(np.mean(np.sum(h * ref, axis=-1))),
                    "unit_feature_mse": float(np.mean((h - ref) ** 2)),
                })

    summary = {
        "protocol": {
            "model": MODEL,
            "device": str(device),
            "windows": len(windows),
            "layers": [3, 6, 12],
            "normalization": "per-window mean removal and peak normalization; unit feature cosine/MSE",
            "warning": "clean-aware proxy, not a phoneme label, WER, or deployment feature",
        },
        "layers": {},
    }
    for layer in (3, 6, 12):
        summary["layers"][str(layer)] = {}
        for arm in ("y_pre", "off", "router"):
            a = np.asarray([r["cosine_to_clean"] for r in out if r["layer"] == layer and r["arm"] == arm])
            m = np.asarray([r["unit_feature_mse"] for r in out if r["layer"] == layer and r["arm"] == arm])
            summary["layers"][str(layer)][arm] = {
                "n": int(len(a)), "cosine_median": float(np.median(a)), "cosine_mean": float(np.mean(a)),
                "mse_median": float(np.median(m)), "mse_mean": float(np.mean(m)),
            }
        for arm in ("off", "router"):
            a = np.asarray([r["cosine_to_clean"] for r in out if r["layer"] == layer and r["arm"] == arm])
            b = np.asarray([r["cosine_to_clean"] for r in out if r["layer"] == layer and r["arm"] == "y_pre"])
            summary["layers"][str(layer)][arm + "_minus_y_pre"] = {
                "cosine_delta_median": float(np.median(a - b)),
                "cosine_delta_mean": float(np.mean(a - b)),
                "cosine_delta_positive": int(np.sum(a > b)),
                "cosine_delta_nonpositive": int(np.sum(a <= b)),
            }
    out_csv = OUT_CSV if args.group == "suppression" else ROOT / "wavlm_all_local_windows.csv"
    out_json = OUT_JSON if args.group == "suppression" else ROOT / "wavlm_all_local_summary.json"
    summary["protocol"]["group"] = args.group
    with out_csv.open("w", newline="") as f:
        fields = list(out[0])
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(out)
    out_json.write_text(json.dumps(summary, indent=2) + "\n")


if __name__ == "__main__":
    main()
