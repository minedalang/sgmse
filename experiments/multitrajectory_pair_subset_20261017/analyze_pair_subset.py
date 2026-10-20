#!/usr/bin/env python3
"""Offline two-trajectory subset analysis; reads existing metric JSON only."""
import itertools
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
SEEDS = [2026, 2027, 2028, 2029]
PAIRS = list(itertools.combinations(SEEDS, 2))
METRICS = ["pesq", "stoi", "sisdr", "dnsmos_p808", "dnsmos_sig", "dnsmos_bak", "dnsmos_ovrl"]
BOOTSTRAP = 20000
BOOTSTRAP_SEED = 2026


def load(path):
    obj = json.loads(path.read_text())
    grouped = {}
    for row in obj["per_trajectory"]:
        grouped.setdefault(row["filename"], {})[int(row["seed"])] = row
    expected = set(SEEDS)
    bad = {name: sorted(set(rows)) for name, rows in grouped.items() if set(rows) != expected}
    if bad:
        raise ValueError(f"incomplete seed records: {bad}")
    return obj, grouped


def ci(deltas, rng):
    n = len(deltas)
    ix = rng.integers(0, n, size=(BOOTSTRAP, n))
    means = deltas[ix].mean(axis=1)
    return [float(np.quantile(means, .025)), float(np.quantile(means, .975))]


def analyze(path):
    source, grouped = load(path)
    names = sorted(grouped)
    baseline = {name: grouped[name][2026] for name in names}
    rows_out = []
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    for pair in PAIRS:
        selected = []
        choose_counts = {str(seed): 0 for seed in pair}
        tie_count = 0
        for name in names:
            a, b = (grouped[name][seed] for seed in pair)
            # Stable tie handling: take the first seed in the listed pair.
            chosen = a if float(a["dnsmos_ovrl"]) >= float(b["dnsmos_ovrl"]) else b
            if float(a["dnsmos_ovrl"]) == float(b["dnsmos_ovrl"]):
                tie_count += 1
            choose_counts[str(chosen["seed"])] += 1
            selected.append(chosen)
        result = {"seed_pair": list(pair), "n_files": len(names),
                  "selected_seed_counts": choose_counts, "ovrl_ties": tie_count,
                  "mean_selected": {}, "vs_seed2026": {},
                  "sisdr_negative_file_count": 0,
                  "sisdr_negative_file_fraction": None}
        sisdr_delta = np.array([float(sel["sisdr"]) - float(baseline[name]["sisdr"])
                                for name, sel in zip(names, selected)])
        result["sisdr_negative_file_count"] = int(np.sum(sisdr_delta < 0))
        result["sisdr_negative_file_fraction"] = float(np.mean(sisdr_delta < 0))
        per_file = []
        for name, sel in zip(names, selected):
            base = baseline[name]
            selected_values = {metric: float(sel[metric]) for metric in METRICS}
            baseline_values = {metric: float(base[metric]) for metric in METRICS}
            per_file.append({
                "filename": name,
                "selected_seed": int(sel["seed"]),
                "selection_ovrl": float(sel["dnsmos_ovrl"]),
                "baseline_seed": 2026,
                "selected": selected_values,
                "baseline": baseline_values,
                "delta_vs_seed2026": {
                    metric: selected_values[metric] - baseline_values[metric]
                    for metric in METRICS
                },
            })
        result["per_file"] = per_file
        for metric in METRICS:
            vals = np.array([float(row[metric]) for row in selected])
            base = np.array([float(baseline[name][metric]) for name in names])
            delta = vals - base
            result["mean_selected"][metric] = float(vals.mean())
            result["vs_seed2026"][metric] = {
                "delta": float(delta.mean()),
                "bootstrap95": ci(delta, rng),
                "baseline_mean": float(base.mean()),
            }
        rows_out.append(result)
    return {"source_file": str(path.relative_to(ROOT)), "n_files": len(names),
            "trajectory_seeds": SEEDS, "trajectory_nfe": 30,
            "pair_compute_cost_score_evaluations_per_input": 60,
            "baseline": "seed 2026, used only for comparison; selection uses DNSMOS OVRL only",
            "selection_rule": "within each seed pair, select higher DNSMOS OVRL; no clean/reference metric used",
            "bootstrap_replicates": BOOTSTRAP, "bootstrap_seed": BOOTSTRAP_SEED,
            "pairs": rows_out}


dev = analyze(ROOT / "experiments/multitrajectory_oracle_20261017/metrics.json")
held = analyze(ROOT / "experiments/multitrajectory_oracle_heldout16/metrics.json")
result = {"experiment_id": "multitrajectory_pair_subset_20261017",
          "method": "offline re-selection from existing four-seed trajectory metrics; no inference or data access",
          "development": dev, "heldout": held}
(OUT / "metrics.json").write_text(json.dumps(result, indent=2) + "\n")

config = {
    "experiment_id": "multitrajectory_pair_subset_20261017",
    "purpose": "Estimate whether two-seed DNSMOS-OVRL selection retains multi-trajectory gains at half of four-trajectory sampling cost.",
    "inputs": ["experiments/multitrajectory_oracle_20261017/metrics.json", "experiments/multitrajectory_oracle_heldout16/metrics.json"],
    "procedure": "For each of the six seed pairs among 2026-2029, select the trajectory with higher DNSMOS OVRL per utterance; compare with seed 2026 using paired utterance bootstrap.",
    "selection_uses_clean_reference": False,
    "new_inference_or_training": False,
    "nfe_per_trajectory": 30,
    "trajectories_per_pair": 2,
    "score_network_evaluations_per_input": 60,
    "bootstrap_replicates": BOOTSTRAP,
    "bootstrap_seed": BOOTSTRAP_SEED,
    "metrics": METRICS,
}
(OUT / "config.yaml").write_text("# YAML-compatible experiment configuration\n" + json.dumps(config, indent=2) + "\n")

lines = ["# Two-trajectory subset selector: offline analysis", "",
         "This reuses the existing four-seed trajectory metrics. It runs no inference or training and reads no clean/reference signal for selection. For each utterance and each of the six seed pairs, it selects the candidate with higher DNSMOS OVRL, then compares the selected output with fixed seed 2026.",
         "", "Each pair costs 2 × N=30 = 60 score-network evaluations per input, half the four-trajectory selector's 120 evaluations. CIs are paired utterance bootstrap intervals (20,000 replicates).", ""]
for split_name, split in [("Development (16)", dev), ("Held-out (16)", held)]:
    lines += [f"## {split_name}", "", "| Seed pair | Selected seed counts | OVRL Δ [95% CI] | SI-SDR Δ dB [95% CI] | SI-SDR negative files | PESQ Δ | STOI Δ | SIG Δ | BAK Δ |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for p in split["pairs"]:
        d = p["vs_seed2026"]
        fmt = lambda m: f"{d[m]['delta']:+.4f} [{d[m]['bootstrap95'][0]:+.4f}, {d[m]['bootstrap95'][1]:+.4f}]"
        lines.append(f"| {p['seed_pair'][0]}–{p['seed_pair'][1]} | {p['selected_seed_counts']} | {fmt('dnsmos_ovrl')} | {fmt('sisdr')} | {p['sisdr_negative_file_count']}/{p['n_files']} | {d['pesq']['delta']:+.4f} | {d['stoi']['delta']:+.4f} | {d['dnsmos_sig']['delta']:+.4f} | {d['dnsmos_bak']['delta']:+.4f} |")
    lines.append("")
lines += ["## Interpretation", "", "The pairwise selector uses an oracle-free DNSMOS criterion but still requires evaluating both trajectories, so it is a compute/quality trade-off rather than a free selector. Treat these as exploratory results: the seed pairs share the same underlying utterances and the DNSMOS criterion is the selection metric, so OVRL gains are expected to be optimistic. Held-out comparisons against fixed seed 2026 are the more informative check; inspect both OVRL and per-utterance SI-SDR regressions before proposing a learned selector or a full test-set run.", "", "Full per-metric means, paired bootstrap intervals, and selection counts are in `metrics.json`."]
(OUT / "experiment_report.md").write_text("\n".join(lines) + "\n")
brief = lambda split: [
    {"pair": p["seed_pair"],
     "ovrl_delta": p["vs_seed2026"]["dnsmos_ovrl"]["delta"],
     "ovrl_ci95": p["vs_seed2026"]["dnsmos_ovrl"]["bootstrap95"],
     "sisdr_negative_files": p["sisdr_negative_file_count"]}
    for p in split["pairs"]
]
print(json.dumps({"output": str(OUT), "development": brief(dev), "heldout": brief(held)}, indent=2))
