#!/usr/bin/env python3
"""Summarize saved posteriors and audit indexing; print JSON or save a new file.

Usage: python3 summary.py [--input PATH] [--output PATH] [--source PATH]
Values describe the existing CSV (including any expanded fixed windows). This
script cannot recover posteriors for corrected windows without saved logits.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter
from pathlib import Path
from statistics import mean


ROOT = Path(__file__).resolve().parent
CONDITIONS = ("clean", "noisy", "y_pre", "off", "router")
# Explicit English IPA consonants, including IPA script g (ɡ), ASCII g, and
# common affricate spellings. No inference from a vowel exclusion list.
CONSONANTS = frozenset((
    "p", "b", "t", "d", "k", "ɡ", "g", "f", "v", "θ", "ð", "s", "z",
    "ʃ", "ʒ", "h", "m", "n", "ŋ", "l", "ɹ", "r", "j", "w", "ɾ", "ʔ",
    "tʃ", "dʒ", "t͡ʃ", "d͡ʒ",
))
METRICS = (
    "fixed_time_target_posterior_mean", "fixed_time_target_top1_fraction",
    "nearby_target_posterior_max", "nearby_target_posterior_mean",
    "nearby_target_top1_fraction",
)


def metric_means(rows):
    return {key: mean(float(row[key]) for row in rows) for key in METRICS}


def summarize(rows):
    groups = sorted({row["group"] for row in rows})
    by_group = {}
    for group in groups:
        arms = {}
        for condition in CONDITIONS:
            selected = [r for r in rows if r["group"] == group and r["condition"] == condition]
            if not selected:
                raise ValueError(f"Missing paired rows: {group}/{condition}")
            arms[condition] = {"n_tokens": len(selected), **metric_means(selected)}
        by_group[group] = arms
    macro, pooled = {}, {}
    if groups:
        for condition in CONDITIONS:
            selected = [r for r in rows if r["condition"] == condition]
            macro[condition] = {
                "n_groups": len(groups),
                **{key: mean(by_group[g][condition][key] for g in groups) for key in METRICS},
            }
            pooled[condition] = {"n_tokens": len(selected), **metric_means(selected)}
    paired = {}
    for label, first, second in (("off_minus_y_pre", "off", "y_pre"),
                                  ("router_minus_off", "router", "off")):
        differences = {
            g: {key: by_group[g][first][key] - by_group[g][second][key] for key in METRICS}
            for g in groups
        }
        paired[label] = {
            "n_paired_groups": len(groups), "by_group": differences,
            "macro_group_mean_difference": {
                key: mean(differences[g][key] for g in groups) for key in METRICS
            } if groups else {},
        }
    return {"per_group": by_group, "macro_group_means": macro,
            "pooled_token_means": pooled, "paired_group_differences": paired}


def audit(rows, source):
    tokens = {(g["group"], int(t["target_index"])): t
              for g in source["groups_detail"] for t in g["clean_target_tokens"]}
    frame_counts = {(a["group"], a["condition"]): int(a["frames"])
                    for a in source["alignments"]}
    seen = set()
    mismatches = {}
    csv_length_disagreements = []
    for row in rows:
        group, condition = row["group"], row["condition"]
        key = (group, int(row["target_index"]))
        identity = (*key, condition)
        if identity in seen:
            raise ValueError(f"Duplicate CSV row: {identity}")
        seen.add(identity)
        token = tokens[key]
        start = int(token["clean_greedy_start_frame"])
        end = int(token["clean_greedy_end_frame_exclusive"])
        assert (int(row["clean_start_frame"]), int(row["clean_end_frame_exclusive"])) == (start, end)
        assert row["token"] == token["token"] and int(row["token_id"]) == int(token["token_id"])
        assert float(row["clean_start_s"]) == token["clean_greedy_start_s"]
        assert float(row["clean_end_s"]) == token["clean_greedy_end_s"]
        lo = max(0, math.floor(token["clean_greedy_start_s"] / 0.02))
        hi = min(frame_counts[(group, condition)],
                 max(lo + 1, math.ceil(token["clean_greedy_end_s"] / 0.02)))
        actual = int(row["fixed_window_frames"])
        if hi - lo != actual:
            csv_length_disagreements.append({"row": identity, "reconstructed": hi - lo, "csv": actual})
        if (lo, hi) != (start, end):
            item = mismatches.setdefault(key, {
                "group": group, "target_index": key[1], "token": token["token"],
                "integer_interval": [start, end], "seconds_interval": [lo, hi],
                "extra_left_frames": start - lo, "extra_right_frames": hi - end,
                "conditions": [], "csv_window_frames": {},
            })
            item["conditions"].append(condition)
            item["csv_window_frames"][condition] = actual
    expected = {(g, i, c) for g, i in tokens for c in CONDITIONS}
    if seen != expected:
        raise ValueError(f"CSV coverage differs: {len(expected - seen)} missing, {len(seen - expected)} extra")
    return {
        "unique_clean_tokens": len(tokens), "token_condition_rows": len(rows),
        "note": "interval_mismatch fields audit the original seconds conversion against source integers; CSV mismatch fields audit actual saved lengths.",
        "interval_mismatch_unique_tokens": len(mismatches),
        "interval_mismatch_rows": sum(len(x["conditions"]) for x in mismatches.values()),
        "csv_length_vs_integer_interval_mismatch_rows": sum(
            int(r["fixed_window_frames"]) != int(r["clean_end_frame_exclusive"]) - int(r["clean_start_frame"])
            for r in rows),
        "csv_length_vs_reconstructed_seconds_interval_disagreements": csv_length_disagreements,
        "mismatched_tokens_per_group": dict(Counter(x["group"] for x in mismatches.values())),
        "mismatches": list(mismatches.values()),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", "--csv", dest="csv", type=Path, default=ROOT / "fixed_time_token_posteriors.csv")
    parser.add_argument("--source", type=Path, default=ROOT.parent / "p8_q_a_ctc_forced_alignment_20261020/metrics.json")
    parser.add_argument("--output", type=Path, help="Write JSON exclusively; refuse an existing file.")
    args = parser.parse_args()
    with args.csv.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    source = json.loads(args.source.read_text())
    selectors = {"all_tokens": lambda t: True,
                 "ipa_consonants": lambda t: t in CONSONANTS,
                 "k_d": lambda t: t in {"k", "d"},
                 "k": lambda t: t == "k", "d": lambda t: t == "d"}
    result = {
        "csv": str(args.csv), "source": str(args.source),
        "interpretation": "Descriptive summaries of existing saved windows; clean-derived pseudo-targets, not verified phoneme truth. No confidence intervals.",
        "weighting": {"macro": "Equal weight per group with eligible tokens.",
                      "pooled": "Equal weight per eligible token; not frame-weighted.",
                      "paired": "Subtract matched condition means within each group, then average groups."},
        "ipa_consonant_inventory": sorted(CONSONANTS),
        "observed_consonants": sorted({r["token"] for r in rows} & CONSONANTS),
        "indexing_audit": audit(rows, source),
        "subsets": {name: summarize([r for r in rows if select(r["token"])])
                    for name, select in selectors.items()},
    }
    payload = json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    if args.output:
        with args.output.open("x") as handle:
            handle.write(payload)
    else:
        print(payload, end="")


if __name__ == "__main__":
    main()
