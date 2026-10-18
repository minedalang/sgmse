#!/usr/bin/env python
"""Generate paper figures from the locked final evaluation only.

This script never runs inference and never reads the raw dataset. It consumes
the already archived per-utterance CSVs and paired-bootstrap JSON.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path("/data/coding/sgmse")
SRC = ROOT / "experiments/final_eval_full_20261018"
OUT = ROOT / "experiments/paper_figures_claim_matrix_20261020"
BOOT = SRC / "metrics/paired_bootstrap.pretty.json"
INTERACTION = ROOT / "experiments/final_eval_reverb_interaction_20261018_metrics.json"
METRICS = [
    "PESQ_e",
    "STOI_e",
    "SI-SDR_e",
    "DNSMOS_SIG_e",
    "DNSMOS_OVRL_e",
    "DNSMOS_P808_e",
    "DNSMOS_BAK_e",
]
LABELS = {
    "PESQ_e": "PESQ",
    "STOI_e": "STOI",
    "SI-SDR_e": "SI-SDR (dB)",
    "DNSMOS_SIG_e": "DNSMOS-SIG",
    "DNSMOS_OVRL_e": "DNSMOS-OVRL",
    "DNSMOS_P808_e": "DNSMOS-P808",
    "DNSMOS_BAK_e": "DNSMOS-BAK",
}


def load_json(path: Path):
    return json.loads(path.read_text())


def style():
    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "figure.dpi": 160,
            "savefig.dpi": 240,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def save_both(fig, stem: str):
    fig.tight_layout()
    fig.savefig(OUT / f"{stem}.png", bbox_inches="tight")
    fig.savefig(OUT / f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)


def delta_by_split(boot):
    fig, axes = plt.subplots(2, 4, figsize=(13, 6.5))
    axes = axes.ravel()
    split_names = [("no_reverb", "No reverb"), ("with_reverb", "With reverb")]
    arms = [("router_vs_off", "Router", "#2166ac"), ("tcut005_vs_off", "t-cut 0.05", "#b2182b")]
    for i, metric in enumerate(METRICS):
        ax = axes[i]
        x = np.arange(2)
        width = 0.35
        for j, (key, label, color) in enumerate(arms):
            vals = [
                boot["splits"][split][key][metric]["mean_delta"]
                for split, _ in split_names
            ]
            ax.bar(x + (j - 0.5) * width, vals, width, label=label, color=color)
        ax.axhline(0, color="black", lw=0.7)
        ax.set_title(LABELS[metric])
        ax.set_xticks(x, [label for _, label in split_names])
        ax.tick_params(axis="x", labelrotation=20)
        ax.grid(axis="y", alpha=0.25)
    axes[-1].axis("off")
    axes[0].legend(frameon=False, loc="best")
    fig.suptitle("Paired change relative to off", y=1.02)
    save_both(fig, "metric_delta_by_split")


def forest_plot(boot, key: str, stem: str, title: str, metrics):
    fig, ax = plt.subplots(figsize=(8, 4.6))
    y = np.arange(len(metrics))
    vals, lows, highs = [], [], []
    for metric in metrics:
        item = boot[key][metric]
        vals.append(item["mean_delta"])
        lows.append(item["ci95"][0])
        highs.append(item["ci95"][1])
    vals, lows, highs = np.array(vals), np.array(lows), np.array(highs)
    ax.errorbar(vals, y, xerr=[vals - lows, highs - vals], fmt="o", color="#2166ac", capsize=3)
    ax.axvline(0, color="black", lw=0.8)
    ax.set_yticks(y, [LABELS[m] for m in metrics])
    ax.invert_yaxis()
    ax.set_xlabel("Paired mean delta (arm − off), 95% bootstrap CI")
    ax.set_title(title)
    ax.grid(axis="x", alpha=0.25)
    save_both(fig, stem)


def interaction_plot(interaction):
    key = "router_interaction_vs_off"
    fig, ax = plt.subplots(figsize=(8, 4.6))
    y = np.arange(len(METRICS))
    vals, lows, highs = [], [], []
    for metric in METRICS:
        item = interaction[key][metric]
        vals.append(item["mean"])
        lows.append(item["ci95"][0])
        highs.append(item["ci95"][1])
    vals, lows, highs = np.array(vals), np.array(lows), np.array(highs)
    ax.errorbar(vals, y, xerr=[vals - lows, highs - vals], fmt="o", color="#762a83", capsize=3)
    ax.axvline(0, color="black", lw=0.8)
    ax.set_yticks(y, [LABELS[m] for m in METRICS])
    ax.invert_yaxis()
    ax.set_xlabel("Reverb interaction: (Router−off)with − (Router−off)without")
    ax.set_title("Router effect changes under reverberation")
    ax.grid(axis="x", alpha=0.25)
    save_both(fig, "reverb_interaction_ci")


def read_csv(arm: str, split: str):
    return pd.read_csv(SRC / f"metrics/{arm}/{split}.csv")


def tradeoff_scatter():
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.7), sharex=True, sharey=True)
    colors = {"no_reverb": "#2166ac", "with_reverb": "#b2182b"}
    for ax, split in zip(axes, ["no_reverb", "with_reverb"]):
        off = read_csv("off", split).set_index("utt")
        router = read_csv("router", split).set_index("utt")
        common = off.index.intersection(router.index)
        x = router.loc[common, "SI-SDR_e"] - off.loc[common, "SI-SDR_e"]
        y = router.loc[common, "DNSMOS_OVRL_e"] - off.loc[common, "DNSMOS_OVRL_e"]
        ax.scatter(x, y, s=13, alpha=0.55, color=colors[split], edgecolors="none")
        ax.axhline(0, color="black", lw=0.7)
        ax.axvline(0, color="black", lw=0.7)
        ax.set_title("No reverb" if split == "no_reverb" else "With reverb")
        ax.set_xlabel("SI-SDR delta (dB)")
        ax.grid(alpha=0.2)
        ax.text(0.03, 0.96, f"n={len(common)}", transform=ax.transAxes, va="top")
    axes[0].set_ylabel("DNSMOS-OVRL delta")
    fig.suptitle("Per-utterance reconstruction–perception trade-off: Router vs off", y=1.02)
    save_both(fig, "sisdr_dnsmos_tradeoff")


def sha256(path: Path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    style()
    boot = load_json(BOOT)
    interaction = load_json(INTERACTION)
    delta_by_split(boot)
    forest_plot(
        boot,
        "router_vs_off",
        "router_off_pooled_ci",
        "Router vs off on the locked 300-utterance evaluation",
        METRICS,
    )
    interaction_plot(interaction)
    tradeoff_scatter()
    summary = {
        "experiment_id": "paper_figures_claim_matrix_20261020",
        "source_bootstrap": str(BOOT),
        "source_bootstrap_sha256": sha256(BOOT),
        "source_interaction": str(INTERACTION),
        "source_interaction_sha256": sha256(INTERACTION),
        "figures": sorted(p.name for p in OUT.glob("*.png")),
        "source_audio_regenerated": False,
        "test_set_tuning": False,
        "dataset_modified": False,
        "checkpoint_modified": False,
        "notes": "Post-hoc visualization of frozen outputs; no bootstrap values recomputed.",
    }
    (OUT / "metrics.json").write_text(json.dumps(summary, indent=2) + "\n")


if __name__ == "__main__":
    main()
