#!/usr/bin/env python3
"""Audit temporal displacement in the fixed-target CTC alignment pilot."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parent
SOURCE = ROOT.parent / "p8_q_a_ctc_forced_alignment_20261020" / "token_alignments.csv"
CONDITIONS = ("clean", "noisy", "y_pre", "off", "router")
CONSONANTS = set("pbtdkgmnfvθðszʃʒhɹljwɾɻ")


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def f(row: dict[str, str], key: str) -> float:
    return float(row[key])


def summary(values: list[float], limit: float) -> dict:
    x = np.asarray(values, dtype=np.float64)
    return {
        "n": int(x.size),
        "median_abs_shift_ms": float(np.median(x)),
        "p90_abs_shift_ms": float(np.quantile(x, 0.90)),
        "within_limit_fraction": float(np.mean(x <= limit)),
        "max_abs_shift_ms": float(np.max(x)),
    }


def main() -> None:
    rows = read_rows(SOURCE)
    clean = {
        (r["group"], r["target_index"], r["token"]): r
        for r in rows
        if r["condition"] == "clean"
    }
    out: list[dict] = []
    for r in rows:
        base = clean[(r["group"], r["target_index"], r["token"])]
        clean_mid = (f(base, "start_s") + f(base, "end_s")) / 2.0
        mid = (f(r, "start_s") + f(r, "end_s")) / 2.0
        shift = (mid - clean_mid) * 1000.0
        out.append({
            "group": r["group"],
            "condition": r["condition"],
            "target_index": int(r["target_index"]),
            "token": r["token"],
            "clean_start_s": f(base, "start_s"),
            "clean_end_s": f(base, "end_s"),
            "aligned_start_s": f(r, "start_s"),
            "aligned_end_s": f(r, "end_s"),
            "shift_ms": shift,
            "abs_shift_ms": abs(shift),
            "consonant": r["token"] in CONSONANTS,
            "target_posterior": f(r, "target_token_posterior_mean"),
        })

    csv_path = ROOT / "boundary_shifts.csv"
    with csv_path.open("w", newline="") as f_out:
        writer = csv.DictWriter(f_out, fieldnames=list(out[0]))
        writer.writeheader()
        writer.writerows(out)

    metrics = {
        "experiment_id": "p8_q_a_ctc_boundary_audit_20261020",
        "status": "completed",
        "source": str(SOURCE),
        "input_rows": len(rows),
        "definition": "absolute midpoint displacement of each condition's forced token path from the clean path, in milliseconds",
        "same_time_claim_limit_ms": 40.0,
        "by_condition": {},
        "by_condition_consonants": {},
    }
    for condition in CONDITIONS:
        subset = [r for r in out if r["condition"] == condition]
        metrics["by_condition"][condition] = summary([r["abs_shift_ms"] for r in subset], 40.0)
        consonants = [r for r in subset if r["consonant"]]
        metrics["by_condition_consonants"][condition] = summary([r["abs_shift_ms"] for r in consonants], 40.0)
    (ROOT / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
