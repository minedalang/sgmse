#!/usr/bin/env python3
"""Offline paired bootstrap for a clean-free trajectory acceptance rule."""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT.parent / "multitrajectory_oracle_heldout16" / "consistency_probe.json"
OUT = ROOT / "metrics.json"


def group_rows(records):
    by = defaultdict(list)
    for row in records:
        by[row["filename"]].append(row)
    return by


def choose(base, winner, min_gain):
    accept = (winner["dnsmos_ovrl"] - base["dnsmos_ovrl"] > min_gain and
              winner["consistency"] <= base["consistency"])
    return winner if accept else base, accept


def deltas(records, min_gain):
    out = []
    for filename, candidates in sorted(group_rows(records).items()):
        base = next(x for x in candidates if x["seed"] == 2026)
        winner = max(candidates, key=lambda x: x["dnsmos_ovrl"])
        selected, accept = choose(base, winner, min_gain)
        out.append({
            "filename": filename,
            "accepted": bool(accept),
            "selected_seed": selected["seed"],
            "ovrl_delta": selected["dnsmos_ovrl"] - base["dnsmos_ovrl"],
            "sisdr_delta": selected["sisdr"] - base["sisdr"],
            "pesq_delta": selected["pesq"] - base["pesq"],
            "stoi_delta": selected["stoi"] - base["stoi"],
            "sig_delta": selected["dnsmos_sig"] - base["dnsmos_sig"],
            "bak_delta": selected["dnsmos_bak"] - base["dnsmos_bak"],
            "winner_ovrl_gain": winner["dnsmos_ovrl"] - base["dnsmos_ovrl"],
            "winner_consistency_delta": winner["consistency"] - base["consistency"],
        })
    return out


def summarize(rows, seed=2026, reps=20000):
    rng = np.random.default_rng(seed)
    metrics = ["ovrl_delta", "sisdr_delta", "pesq_delta", "stoi_delta", "sig_delta", "bak_delta"]
    result = {"n_files": len(rows), "accepted": sum(x["accepted"] for x in rows),
              "sisdr_negative_files": sum(x["sisdr_delta"] < 0 for x in rows)}
    for metric in metrics:
        x = np.asarray([r[metric] for r in rows], dtype=float)
        boot = np.asarray([x[rng.integers(0, len(x), len(x))].mean() for _ in range(reps)])
        result[metric] = {"mean": float(x.mean()), "bootstrap95": [float(np.quantile(boot, .025)), float(np.quantile(boot, .975))]}
    result["worst_sisdr_delta"] = float(min(x["sisdr_delta"] for x in rows))
    result["rows"] = rows
    return result


def main():
    source = json.loads(SOURCE.read_text())
    result = {"experiment_id": "multitrajectory_consistency_gain_guard_20261020",
              "source": str(SOURCE), "selection_uses_clean": False,
              "rule": "max OVRL candidate accepted iff OVRL gain >= threshold and consistency <= seed-2026 consistency",
              "thresholds": {}}
    for group in ("dev", "heldout"):
        records = source[group]
        result["thresholds"][group] = {}
        for threshold in (0.0, 0.02, 0.05, 0.08):
            result["thresholds"][group][str(threshold)] = summarize(deltas(records, threshold))
    OUT.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
