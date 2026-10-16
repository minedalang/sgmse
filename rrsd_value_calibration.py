#!/usr/bin/env python3
"""Prepare RRSD value-calibration tasks from an existing t-cut screen.

This utility only prepares manifests and configuration.  It deliberately does
not load a model, run inference, or evaluate audio.
"""

from __future__ import annotations

import argparse
import csv
import itertools
import inspect
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable


DEFAULT_MANIFEST = Path("experiment_logs/rrsd_tcut_screen/manifest.csv")
DEFAULT_PAIRED = Path("experiment_logs/rrsd_tcut_screen/paired_metrics.csv")


def _csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _safe_output_dir(base: Path) -> Path:
    """Return a fresh output directory; never overwrite an existing run."""
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    candidate = base / f"rrsd_value_calibration_{stamp}"
    suffix = 0
    while candidate.exists():
        suffix += 1
        candidate = base / f"rrsd_value_calibration_{stamp}_{suffix:02d}"
    return candidate


def _yaml_value(value: object) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if value is None:
        return "null"
    if isinstance(value, (int, float)):
        return str(value)
    return json.dumps(str(value), ensure_ascii=False)


def _write_yaml(path: Path, config: dict[str, object]) -> None:
    lines: list[str] = []
    for key, value in config.items():
        if isinstance(value, list):
            lines.append(f"{key}:")
            for item in value:
                lines.append(f"  - {_yaml_value(item)}")
        else:
            lines.append(f"{key}: {_yaml_value(value)}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _run_smoke(tasks: list[dict[str, object]]) -> None:
    """Perform a dependency/API smoke check for one candidate.

    Imports of torch and the model are intentionally kept here.  The normal
    manifest and dry-run paths therefore remain usable in lightweight
    environments.  A full candidate probe is not attempted: the enhancement
    entry point currently hard-codes CUDA and running it here would turn a
    smoke check into an inference job.
    """
    try:
        import torch  # noqa: F401  # delayed by design
        from sgmse.joint_model import JointSemambaSgmseModel  # delayed by design
        from sgmse import sampling
    except Exception as exc:
        print(f"SMOKE UNAVAILABLE: could not import torch/model ({type(exc).__name__}: {exc})")
        return

    task = tasks[0]
    checkpoint = str(task.get("checkpoint", ""))
    noisy = str(task.get("noisy", ""))
    if not checkpoint or checkpoint.startswith("<"):
        print("SMOKE UNAVAILABLE: first candidate has no concrete checkpoint path.")
        return
    if not noisy:
        print("SMOKE UNAVAILABLE: first candidate has no noisy input path.")
        return
    checkpoint_path = Path(checkpoint).expanduser()
    noisy_path = Path(noisy).expanduser()
    if not checkpoint_path.is_file():
        print(f"SMOKE UNAVAILABLE: checkpoint not found: {checkpoint_path}")
        return
    if not noisy_path.is_file():
        print(f"SMOKE UNAVAILABLE: noisy input not found: {noisy_path}")
        return

    # The sampler is the actual trace boundary used by candidate_probe.  Do
    # not claim a probe result unless that interface is present.
    sampler_signature = inspect.signature(sampling.get_pc_sampler)
    model_signature = inspect.signature(JointSemambaSgmseModel.enhance)
    has_trace = "trace_sink" in sampler_signature.parameters or any(
        p.kind is inspect.Parameter.VAR_KEYWORD
        for p in sampler_signature.parameters.values()
    )
    if not has_trace:
        print("SMOKE UNAVAILABLE: candidate_probe requires sampler trace_sink interface.")
        return
    if not any(p.kind is inspect.Parameter.VAR_KEYWORD for p in model_signature.parameters.values()):
        print("SMOKE UNAVAILABLE: model.enhance does not forward trace options.")
        return
    print(f"SMOKE READY: candidate_probe trace interface available for {noisy_path.name}.")
    print("SMOKE SKIPPED: full candidate_probe is not run by this CPU-safe smoke check; no metrics fabricated.")


def build_tasks(manifest_rows: Iterable[dict[str, str]], metric_rows: Iterable[dict[str, str]],
                alphas: list[float], k_steps: list[int], seeds: list[int],
                checkpoints: list[str]) -> list[dict[str, object]]:
    metric_index = {(row.get("filename", ""), row.get("uttid", "")): row
                    for row in metric_rows}
    tasks: list[dict[str, object]] = []
    for row in manifest_rows:
        key = (row.get("filename", ""), row.get("uttid", ""))
        if key not in metric_index:
            raise ValueError(f"No paired_metrics.csv row for filename={key[0]!r}, uttid={key[1]!r}")
        for alpha, k_step, seed, checkpoint in itertools.product(
                alphas, k_steps, seeds, checkpoints):
            task_id = f"{row['uttid']}_a{alpha:g}_k{k_step}_s{seed}"
            tasks.append({
                "task_id": task_id,
                "filename": row.get("filename", ""),
                "uttid": row.get("uttid", ""),
                "noisy": row.get("noisy", ""),
                "clean": row.get("clean", ""),
                "alpha": alpha,
                "k_steps": k_step,
                "seed": seed,
                "checkpoint": checkpoint,
                "source_metric_row": f"{key[0]}::{key[1]}",
            })
    return tasks


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare RRSD value-calibration manifest/config without running a model.")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST,
                        help=f"Input screen manifest CSV (default: {DEFAULT_MANIFEST})")
    parser.add_argument("--paired_metrics", type=Path, default=None,
                        help="Input paired_metrics.csv (default: sibling of --manifest)")
    parser.add_argument("--out_dir", type=Path, default=Path("experiment_logs"),
                        help="Parent directory for a new calibration run")
    parser.add_argument("--alpha", nargs="+", type=float, default=[0.25, 0.5, 0.75],
                        help="Candidate alpha values")
    parser.add_argument("--k_step", nargs="+", type=int, default=[1, 2, 4],
                        help="Candidate K-step values")
    parser.add_argument("--seed", nargs="+", type=int, default=[2026],
                        help="Candidate random seeds")
    parser.add_argument("--checkpoint", nargs="+", default=["<set-checkpoint-path>"],
                        help="Checkpoint placeholders or paths")
    parser.add_argument("--dry_run", "--dry-run", dest="dry_run", action="store_true",
                        help="Validate inputs and print the planned outputs without writing files")
    parser.add_argument("--run_smoke", "--run-smoke", action="store_true",
                        help="Run a CPU-safe single-candidate trace/API smoke check (no training or metrics)")
    args = parser.parse_args()

    manifest_path = args.manifest.expanduser().resolve()
    paired_path = (args.paired_metrics or manifest_path.with_name("paired_metrics.csv")).expanduser().resolve()
    if not manifest_path.is_file():
        parser.error(f"manifest not found: {manifest_path}")
    if not paired_path.is_file():
        # Preserve the requested default location when --manifest points at a custom tree.
        if args.paired_metrics is None and manifest_path.parent == DEFAULT_MANIFEST.parent.resolve():
            paired_path = DEFAULT_PAIRED.resolve()
        if not paired_path.is_file():
            parser.error(f"paired metrics not found: {paired_path}")

    manifest_rows = _csv_rows(manifest_path)
    metric_rows = _csv_rows(paired_path)
    tasks = build_tasks(manifest_rows, metric_rows, args.alpha, args.k_step, args.seed, args.checkpoint)
    if not tasks:
        parser.error("input manifest contains no rows")

    output_dir = _safe_output_dir(args.out_dir.expanduser().resolve())
    config = {
        "schema_version": 1,
        "purpose": "rrsd_value_calibration_task_manifest",
        "source_manifest": str(manifest_path),
        "source_paired_metrics": str(paired_path),
        "task_manifest": "calibration_manifest.csv",
        "alpha_candidates": args.alpha,
        "k_step_candidates": args.k_step,
        "seed_candidates": args.seed,
        "checkpoint_candidates": args.checkpoint,
        "task_count": len(tasks),
        "run_inference": False,
        "run_evaluation": False,
    }
    print(f"Validated {len(manifest_rows)} manifest rows and {len(metric_rows)} paired metric rows.")
    print(f"Planned {len(tasks)} calibration tasks.")
    print(f"Output directory: {output_dir}")
    if args.dry_run:
        print("Dry run: no files written.")
        return

    if args.run_smoke:
        _run_smoke(tasks)

    output_dir.mkdir(parents=True, exist_ok=False)
    manifest_out = output_dir / "calibration_manifest.csv"
    config_out = output_dir / "config.yaml"
    with manifest_out.open("w", newline="", encoding="utf-8") as handle:
        fields = list(tasks[0])
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(tasks)
    _write_yaml(config_out, config)
    print(f"Wrote {manifest_out}")
    print(f"Wrote {config_out}")


if __name__ == "__main__":
    main()
