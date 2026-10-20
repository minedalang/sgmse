#!/usr/bin/env python3
"""Read-only Distill-MOS audit of existing multi-trajectory WAVs."""
from __future__ import annotations

import csv
import hashlib
import json
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from scipy.stats import pearsonr, spearmanr
import distillmos

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT.parent / "multitrajectory_oracle_heldout16" / "consistency_probe.json"
OUT_CSV = ROOT / "per_trajectory.csv"
OUT_JSON = ROOT / "metrics.json"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def corr(a, b):
    if np.std(a) == 0 or np.std(b) == 0:
        return {"pearson": float("nan"), "spearman": float("nan")}
    return {"pearson": float(pearsonr(a, b).statistic), "spearman": float(spearmanr(a, b).statistic)}


def main():
    source = json.loads(SOURCE.read_text())
    model = distillmos.ConvTransformerSQAModel().eval().to(DEVICE)
    start = time.perf_counter()
    out = []
    for group in ("dev", "heldout"):
        for row in source[group]:
            wav_path = Path(row["wav"])
            if not wav_path.exists():
                raise FileNotFoundError(wav_path)
            wav, sr = sf.read(wav_path, dtype="float32", always_2d=False)
            if sr != 16000:
                raise RuntimeError((wav_path, sr))
            x = torch.from_numpy(np.asarray(wav, dtype=np.float32))[None].to(DEVICE)
            with torch.inference_mode():
                score = float(model(x).reshape(-1)[0].item())
            out.append({**row, "group": group, "distillmos": score})

    by = defaultdict(list)
    for row in out:
        by[(row["group"], row["filename"])].append(row)
    selected = []
    for (group, filename), rows in sorted(by.items()):
        base = next(r for r in rows if r["seed"] == 2026)
        winner = max(rows, key=lambda r: r["dnsmos_ovrl"])
        guard = winner if (winner["dnsmos_ovrl"] > base["dnsmos_ovrl"] and winner["consistency"] <= base["consistency"]) else base
        for name, r in (("baseline", base), ("ovrl_winner", winner), ("consistency_guard", guard)):
            selected.append({
                "group": group, "filename": filename, "selection": name, "selected_seed": r["seed"],
                "distillmos": r["distillmos"],
                "distill_delta_vs_seed2026": r["distillmos"] - base["distillmos"],
                "ovrl_delta_vs_seed2026": r["dnsmos_ovrl"] - base["dnsmos_ovrl"],
                "sisdr_delta_vs_seed2026": r["sisdr"] - base["sisdr"],
                "pesq_delta_vs_seed2026": r["pesq"] - base["pesq"],
                "stoi_delta_vs_seed2026": r["stoi"] - base["stoi"],
            })

    summary = {
        "experiment_id": "multitrajectory_distillmos_audit_20261020",
        "device": str(DEVICE), "n_trajectory_wavs": len(out), "runtime_seconds": time.perf_counter() - start,
        "new_inference": False, "new_training": False,
        "selections": {},
        "limitations": [
            "Distill-MOS is another learned proxy, not human MOS or transcription.",
            "The trajectory WAVs and OVRL scores were generated previously; no fresh stochastic samples were made.",
            "Selection and evaluation remain exploratory and are not a deployable claim.",
        ],
    }
    fields = list(out[0])
    with (ROOT / "per_trajectory.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(out)
    fields = list(selected[0])
    with (ROOT / "selected_summary.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(selected)
    for group in ("dev", "heldout"):
        summary["selections"][group] = {}
        for selection in ("baseline", "ovrl_winner", "consistency_guard"):
            rows = [r for r in selected if r["group"] == group and r["selection"] == selection]
            summary["selections"][group][selection] = {
                "n": len(rows),
                "accepted_nonbaseline": sum(r["selected_seed"] != 2026 for r in rows),
                "means": {k: float(np.mean([r[k] for r in rows])) for k in ("distill_delta_vs_seed2026", "ovrl_delta_vs_seed2026", "sisdr_delta_vs_seed2026", "pesq_delta_vs_seed2026", "stoi_delta_vs_seed2026")},
                "negative_sisdr_files": sum(r["sisdr_delta_vs_seed2026"] < 0 for r in rows),
                "correlation_distill_vs_ovrl": corr([r["distill_delta_vs_seed2026"] for r in rows], [r["ovrl_delta_vs_seed2026"] for r in rows]),
                "correlation_distill_vs_sisdr": corr([r["distill_delta_vs_seed2026"] for r in rows], [r["sisdr_delta_vs_seed2026"] for r in rows]),
            }
    OUT_JSON.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
