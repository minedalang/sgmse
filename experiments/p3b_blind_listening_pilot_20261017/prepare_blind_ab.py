#!/usr/bin/env python3
"""Prepare a reproducible, blinded exploratory A/B listening pilot.

This script only reads the existing manifest and three already-generated
enhancement directories. It never reads or writes files under the dataset.
"""

from __future__ import annotations

import csv
import hashlib
import html
import json
import random
import re
import shutil
import wave
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
MANIFEST = ROOT / "experiment_logs/rrsd_tcut_screen/manifest.csv"
ARMS = {
    "router": ROOT / "experiment_logs/rrsd_tcut_screen/router",
    "cut005": ROOT / "experiment_logs/rrsd_tcut_screen/cut005",
    "selective": ROOT / "experiment_logs/rrsd_selective_screen/selective",
}
SEED = 20261017
BIN_RANGES = (
    ("very_low", -100, -16),
    ("low", -15, -12),
    ("mid", -11, -9),
    ("higher", -8, 100),
)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read_wav_header(path: Path) -> tuple[int, int, int, int]:
    with wave.open(str(path), "rb") as f:
        return f.getframerate(), f.getnchannels(), f.getsampwidth(), f.getnframes()


def get_snr(path: str) -> int:
    match = re.search(r"_snr(-?\d+(?:\.\d+)?)_", Path(path).name)
    if not match:
        raise ValueError(f"Could not parse nominal SNR from noisy filename: {path}")
    return int(float(match.group(1)))


def choose_utterances(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    rng = random.Random(SEED)
    selected: list[dict[str, str]] = []
    used_uttids: set[str] = set()
    for bin_name, low, high in BIN_RANGES:
        candidates = [
            row for row in rows
            if low <= get_snr(row["noisy"]) <= high
            and row["uttid"] not in used_uttids
        ]
        candidates.sort(key=lambda row: row["filename"])
        if len(candidates) < 2:
            raise RuntimeError(f"Need two utterances in SNR stratum {bin_name}")
        chosen = rng.sample(candidates, 2)
        selected.extend(chosen)
        used_uttids.update(row["uttid"] for row in chosen)
    return selected


def main() -> None:
    if OUT.exists():
        existing = {path.name for path in OUT.iterdir() if path.name != Path(__file__).name}
        if existing:
            raise FileExistsError(
                f"Refusing to overwrite existing experiment artifacts in {OUT}: {sorted(existing)}"
            )
    if set(ARMS) != {"router", "cut005", "selective"}:
        raise AssertionError("Unexpected arm set")

    with MANIFEST.open(newline="", encoding="utf-8") as f:
        all_rows = list(csv.DictReader(f))
    rows_by_filename = {row["filename"]: row for row in all_rows}
    arm_names = list(ARMS)
    filename_sets = {
        arm: {p.name for p in directory.glob("*.wav")}
        for arm, directory in ARMS.items()
    }
    common = set.intersection(*(filename_sets[arm] for arm in arm_names))
    expected = set(rows_by_filename)
    if len(all_rows) != 32 or common != expected:
        raise RuntimeError(
            f"Expected exactly 32 manifest-matched files in all arms; "
            f"manifest={len(expected)}, common={len(common)}"
        )

    selected = choose_utterances(all_rows)
    rng = random.Random(SEED)
    pairs = [("router", "cut005"), ("router", "selective"), ("cut005", "selective")]
    trials: list[dict[str, str]] = []
    for row in selected:
        for left_arm, right_arm in pairs:
            if rng.random() < 0.5:
                left_arm, right_arm = right_arm, left_arm
            trials.append({
                "filename": row["filename"],
                "uttid": row["uttid"],
                "nominal_snr_db": str(get_snr(row["noisy"])),
                "left_arm": left_arm,
                "right_arm": right_arm,
            })
    rng.shuffle(trials)

    (OUT / "playback" / "audio").mkdir(parents=True, exist_ok=True)
    (OUT / "scoring").mkdir(parents=True, exist_ok=True)
    key_rows = []
    playback_rows = []
    total_bytes = 0
    reference_header = None
    for idx, trial in enumerate(trials, 1):
        trial_id = f"T{idx:02d}"
        wavs = {}
        for side, arm in (("A", trial["left_arm"]), ("B", trial["right_arm"])):
            source = ARMS[arm] / trial["filename"]
            if not source.is_file():
                raise FileNotFoundError(source)
            header = read_wav_header(source)
            if reference_header is None:
                reference_header = header
            if header != reference_header:
                raise RuntimeError(f"Mismatched WAV format/duration for {source}: {header}")
            target = OUT / "playback" / "audio" / f"{trial_id}_{side}.wav"
            shutil.copyfile(source, target)
            total_bytes += target.stat().st_size
            wavs[side] = {"path": str(source.relative_to(ROOT)), "sha256": sha256(source)}
        playback_rows.append({"trial_id": trial_id})
        key_rows.append({
            "trial_id": trial_id,
            "filename": trial["filename"],
            "uttid": trial["uttid"],
            "nominal_snr_db": trial["nominal_snr_db"],
            "A_condition": trial["left_arm"],
            "B_condition": trial["right_arm"],
            "A_source": wavs["A"]["path"],
            "B_source": wavs["B"]["path"],
            "A_sha256": wavs["A"]["sha256"],
            "B_sha256": wavs["B"]["sha256"],
        })

    with (OUT / "scoring" / "answer_key.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(key_rows[0]))
        writer.writeheader()
        writer.writerows(key_rows)
    with (OUT / "responses.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["trial_id", "preference_A_B_tie", "strength_1_2", "artifact_or_naturalness_notes"],
        )
        writer.writeheader()
        for row in playback_rows:
            writer.writerow({"trial_id": row["trial_id"]})

    cards = []
    for row in playback_rows:
        tid = row["trial_id"]
        cards.append(
            f"<section class='trial'><h2>{tid}</h2>"
            f"<p>Listen to both versions once or twice. Choose the more natural / less distorted version, or tie.</p>"
            f"<div class='side'><h3>A</h3><audio controls preload='none' src='audio/{tid}_A.wav'></audio></div>"
            f"<div class='side'><h3>B</h3><audio controls preload='none' src='audio/{tid}_B.wav'></audio></div>"
            f"<p>Record the choice in <code>responses.csv</code>; use A, B, or tie. Strength: 1=slight, 2=clear.</p></section>"
        )
    page = """<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Blind A/B listening pilot</title>
<style>body{font:16px system-ui,sans-serif;max-width:850px;margin:2rem auto;padding:0 1rem;background:#fafafa;color:#222}.trial{background:white;border:1px solid #ddd;border-radius:8px;padding:1rem;margin:1rem 0}.side{margin:.8rem 0}audio{width:100%}h2{margin:.2rem 0}.note{background:#fff4ce;padding:1rem;border-radius:8px}</style>
</head><body><h1>Blind A/B listening pilot</h1>
<div class="note"><strong>Exploratory pilot:</strong> 8 utterances from a repeatedly used development set, rated by one listener. This is for qualitative diagnosis only, not an independent test or a population-level claim. Do not open the separate scoring key before finishing ratings. Keep playback volume fixed; do not loudness-normalize either side.</div>
<p>For each trial, compare A and B and select the one that sounds more natural and less distorted overall, or choose tie/no preference. Consider residual noise, metallic/robotic sound, interruptions, timbre, and intelligibility. Do not try to guess which system produced a version. Record one response per trial.</p>
""" + "\n".join(cards) + "\n</body></html>"
    (OUT / "playback" / "index.html").write_text(page, encoding="utf-8")

    snr_counts: dict[str, int] = {}
    for row in selected:
        stratum = next(name for name, lo, hi in BIN_RANGES if lo <= get_snr(row["noisy"]) <= hi)
        snr_counts[stratum] = snr_counts.get(stratum, 0) + 1
    config = {
        "schema_version": 1,
        "experiment_id": OUT.name,
        "status": "prepared_pending_listener_ratings",
        "purpose": "exploratory_blinded_paired_preference_pilot",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "random_seed": SEED,
        "source_manifest": str(MANIFEST.relative_to(ROOT)),
        "source_arms": {arm: str(path.relative_to(ROOT)) for arm, path in ARMS.items()},
        "source_conditions": {
            "router": "RRSD scale=1.0; PC sampler N=30; seed=2026",
            "cut005": "RRSD scale=1.0; t_cut=0.05; PC sampler N=30; seed=2026",
            "selective": "RRSD scale=1.0; 0.15<=t<=0.75; PC sampler N=30; seed=2026",
        },
        "selected_utterances": len(selected),
        "trials": len(trials),
        "pairwise_comparisons_per_utterance": pairs,
        "snr_strata_counts": snr_counts,
        "audio_format": {
            "sample_rate_hz": reference_header[0],
            "channels": reference_header[1],
            "sample_width_bytes": reference_header[2],
            "frames_per_clip": reference_header[3],
        },
        "playback_audio_bytes": total_bytes,
        "dataset_modified": False,
        "inference_or_training_run": False,
        "source_audio_copied_from": "existing generated experiment outputs only",
        "human_rating_scope": "single-listener exploratory pilot; no inferential/generalization claim",
    }
    (OUT / "config.yaml").write_text(
        "\n".join([
            f"schema_version: {config['schema_version']}",
            f"experiment_id: {config['experiment_id']}",
            f"status: {config['status']}",
            f"purpose: {config['purpose']}",
            f"random_seed: {SEED}",
            f"source_manifest: {config['source_manifest']}",
            "source_conditions:",
            "  router: RRSD scale=1.0; PC N=30; seed=2026",
            "  cut005: RRSD scale=1.0; t_cut=0.05; PC N=30; seed=2026",
            "  selective: RRSD scale=1.0; 0.15<=t<=0.75; PC N=30; seed=2026",
            "sampling:",
            "  utterances: 8 (2 per nominal-SNR stratum)",
            "  pairwise_trials: 24 (all three pairs among the three conditions)",
            f"  snr_strata_counts: {json.dumps(snr_counts, sort_keys=True)}",
            f"audio_format: {json.dumps(config['audio_format'], sort_keys=True)}",
            f"playback_audio_bytes: {total_bytes}",
            "dataset_modified: false",
            "inference_or_training_run: false",
            "ratings: single-listener exploratory pilot; not independent test evidence",
        ]) + "\n",
        encoding="utf-8",
    )
    metrics = {
        "status": "pending_listener_ratings",
        "num_utterances": len(selected),
        "num_pairwise_trials": len(trials),
        "rating_counts": {"A": None, "B": None, "tie": None},
        "per_condition_pair_counts": {"router_vs_cut005": None, "router_vs_selective": None, "cut005_vs_selective": None},
        "inferential_statistics": None,
        "note": "No subjective outcomes are recorded yet. Do not interpret this preparation artifact as an experiment result.",
    }
    (OUT / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    (OUT / "experiment_report.md").write_text(
        f"""# P3b blinded listening pilot package

## Status

Prepared for listening; no human ratings have been collected. This is an exploratory pilot, not a completed preference experiment.

## Question and design

Can a listener consistently distinguish the existing Router, `t_cut=0.05`, and selective-window outputs on matched utterances, and which version sounds more natural / less distorted? The pilot uses 8 utterances selected reproducibly from four nominal-SNR strata (2 each) of the existing 32-utterance development manifest. For each utterance, all three condition pairs are presented once, giving 24 trials. Side assignment and trial order are randomized with seed `{SEED}`.

## Matched inputs and provenance

The three existing output directories contain the same 32 manifest filenames. Their inference setup is documented as the same 80k EMA checkpoint, per-file seed 2026, PC sampler, `N=30`, `snr=0.5`, no corrector. Conditions are Router scale 1, `t_cut=0.05` scale 1, and selective scale 1 over `0.15 <= t <= 0.75`. The Router and t-cut commands are recorded in `experiment_logs/rrsd_tcut_screen/*.log`; selective setup is in `experiments/rrsd_selective_screen/config.yaml` and its report.

Only existing generated WAVs were copied into `playback/audio/` under anonymous trial/side filenames. No dataset file, checkpoint, model, or inference output was modified; no inference, training, or GPU work was run. The answer key and source checksums are in `scoring/answer_key.csv`; do not inspect it until ratings are complete. `playback/index.html` contains the blind player, and `responses.csv` is the response template.

## Rating procedure

Use headphones in a quiet setting. Fix playback volume before starting and leave it unchanged; do not loudness-normalize the files. For each trial, choose A, B, or tie/no preference based on overall naturalness and distortion, optionally noting residual noise, metallic/robotic sound, discontinuities, timbre, or intelligibility. “Strength 1” means a slight preference and “2” a clear preference. If uncertain, use tie. Do not guess the system identity.

## Limitations and interpretation

This is one listener and eight utterances from a repeatedly used development set. Utterances are reused across three comparisons, so trials are not independent. Results can identify obvious, repeatable listening issues and guide the next experiment, but cannot establish general human preference, independent generalization, statistical significance, or paper-level perceptual improvement. If this pilot is informative, repeat with independent listeners and a newly locked set before selecting a training target.

## Current artifacts

- `config.yaml`: fixed selection, pair randomization, and source conditions.
- `metrics.json`: pending outcome schema; contains no measured ratings.
- `playback/index.html` and `playback/audio/`: anonymized listening material.
- `responses.csv`: fill-in response sheet.
- `scoring/answer_key.csv`: condition/file mapping and source hashes; open only after rating.
- `prepare_blind_ab.py`: reproducible package-generation script.
""",
        encoding="utf-8",
    )

    print(json.dumps({
        "output": str(OUT),
        "selected_files": [row["filename"] for row in selected],
        "trials": len(trials),
        "copied_bytes": total_bytes,
        "audio_format": config["audio_format"],
        "snr_strata_counts": snr_counts,
    }, indent=2))


if __name__ == "__main__":
    main()
