#!/usr/bin/env python
"""Recompute frozen final-evaluation means from per-utterance CSVs."""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path


ROOT = Path("/data/coding/sgmse")
SRC = ROOT / "experiments/final_eval_full_20261018"
OUT = ROOT / "experiments/final_metrics_repro_audit_20261020"
METRICS = ["PESQ_e", "STOI_e", "SI-SDR_e", "DNSMOS_SIG_e", "DNSMOS_OVRL_e", "DNSMOS_P808_e", "DNSMOS_BAK_e"]
ARMS = ["off", "router", "tcut005"]
SPLITS = ["no_reverb", "with_reverb"]


def read_csv(arm: str, split: str):
    with (SRC / f"metrics/{arm}/{split}.csv").open(newline="") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 150, (arm, split, len(rows))
    return rows


def mean(rows, key):
    return sum(float(r[key]) for r in rows) / len(rows)


def sha(path: Path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rows = {(arm, split): read_csv(arm, split) for arm in ARMS for split in SPLITS}
    source = json.loads((SRC / "metrics/paired_bootstrap.pretty.json").read_text())
    result = {
        "experiment_id": "final_metrics_repro_audit_20261020",
        "source_csv_sha256": {
            f"{arm}/{split}": sha(SRC / f"metrics/{arm}/{split}.csv")
            for arm in ARMS for split in SPLITS
        },
        "recomputed_means": {},
        "max_abs_mean_error": 0.0,
        "checks": {},
    }
    for split in SPLITS:
        result["recomputed_means"][split] = {}
        for arm in ARMS:
            result["recomputed_means"][split][arm] = {}
            for metric in METRICS:
                value = mean(rows[(arm, split)], metric)
                reference = source["splits"][split]["arms"][arm][metric]
                error = abs(value - reference)
                result["max_abs_mean_error"] = max(result["max_abs_mean_error"], error)
                assert error < 2e-10, (split, arm, metric, value, reference, error)
                result["recomputed_means"][split][arm][metric] = value
    result["checks"]["all_rows_150"] = True
    result["checks"]["per_split_paired_utt_ids"] = True
    for split in SPLITS:
        ids = {arm: [r["utt"] for r in rows[(arm, split)] for _ in [0]] for arm in ARMS}
        assert len(set(ids["off"])) == 150
        assert ids["off"] == ids["router"] == ids["tcut005"]
    result["checks"]["pooled_router_deltas_match"] = True
    result["pooled_router_deltas"] = {}
    for metric in METRICS:
        off = [r for split in SPLITS for r in rows[("off", split)]]
        router = [r for split in SPLITS for r in rows[("router", split)]]
        delta = sum(float(b[metric]) - float(a[metric]) for a, b in zip(off, router)) / len(off)
        reference = source["router_vs_off"][metric]["mean_delta"]
        assert abs(delta - reference) < 2e-10, (metric, delta, reference)
        result["pooled_router_deltas"][metric] = delta
    (OUT / "metrics.json").write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
