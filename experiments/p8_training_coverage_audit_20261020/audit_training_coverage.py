#!/usr/bin/env python3
"""Read-only audit of the DNS-Ins20 synthetic train/CV coverage.

The script reads only the train_synthetic and cv_synthetic manifests and
sound-file headers. It does not instantiate the training data loader, modify
the dataset, read the frozen test manifest, or start model inference.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import statistics
import subprocess
from collections import Counter, defaultdict
from pathlib import Path

import soundfile as sf


SNR_RE = re.compile(r"(?:^|_)snr(-?\d+)(?:_|\.wav$)", re.IGNORECASE)
SPK_RE = re.compile(r"reader_(\d+)", re.IGNORECASE)
FILEID_RE = re.compile(r"fileid_(\d+)", re.IGNORECASE)
KNOWN_NOISE_TOKENS = (
    "typing",
    "door",
    "fan",
    "squeak",
    "squeakychair",
    "breath",
    "spit",
    "keyboard",
    "paper",
    "clap",
    "music",
    "cough",
    "laugh",
    "click",
    "bell",
    "car",
    "traffic",
)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def git_info(repo: Path) -> dict:
    def run(*args: str) -> str | None:
        try:
            return subprocess.check_output(["git", "-C", str(repo), *args], text=True).strip()
        except Exception:
            return None

    return {
        "commit": run("rev-parse", "HEAD"),
        "status_porcelain": run("status", "--short"),
    }


def read_scp(path: Path) -> list[tuple[str, str]]:
    rows: list[tuple[str, str]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, raw in enumerate(f, 1):
            line = raw.strip()
            if not line:
                continue
            fields = line.split(maxsplit=1)
            if len(fields) != 2:
                raise ValueError(f"bad {path}:{line_no}: {raw!r}")
            rows.append((fields[0], fields[1]))
    return rows


def resolve_path(scp: Path, value: str) -> Path:
    p = Path(value)
    return p if p.is_absolute() else (scp.parent / p).resolve()


def parse_snr(path: Path) -> int | None:
    m = SNR_RE.search(path.name)
    return int(m.group(1)) if m else None


def parse_fileid(utt: str, path: Path) -> str | None:
    m = FILEID_RE.search(utt) or FILEID_RE.search(path.name)
    return m.group(1) if m else None


def parse_speaker(utt: str, path: Path) -> str | None:
    m = SPK_RE.search(utt) or SPK_RE.search(path.name)
    return m.group(1) if m else None


def filename_noise_tokens(path: Path) -> list[str]:
    lower = path.name.lower()
    found = []
    for token in KNOWN_NOISE_TOKENS:
        if token in lower:
            found.append(token)
    return sorted(set(found)) or ["unknown_or_unlabelled"]


def priority_bin(snr: int | None) -> str:
    if snr is None:
        return "unknown"
    if -20 <= snr < -18:
        return "[-20,-18)"
    if -18 <= snr < -15:
        return "[-18,-15)"
    if -15 <= snr < -10:
        return "[-15,-10)"
    if -10 <= snr <= -5:
        return "[-10,-5]"
    return "outside_requested_range"


def summary_numbers(values: list[float]) -> dict:
    if not values:
        return {"n": 0}
    vals = sorted(values)
    return {
        "n": len(vals),
        "sum": float(sum(vals)),
        "mean": float(statistics.fmean(vals)),
        "min": float(vals[0]),
        "p50": float(statistics.median(vals)),
        "max": float(vals[-1]),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset-root", type=Path, required=True)
    ap.add_argument("--manifest-root", type=Path, required=True)
    ap.add_argument("--repo", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    out = args.out
    out.mkdir(parents=True, exist_ok=True)

    records_by_split: dict[str, list[dict]] = {}
    file_hashes: dict[str, str] = {}
    global_failures: list[dict] = []
    for split in ("tr_synthetic", "cv_synthetic"):
        split_dir = args.manifest_root / split
        paths = {name: split_dir / f"{name}.scp" for name in ("wav", "spk1", "noise1")}
        for p in paths.values():
            if not p.is_file():
                raise FileNotFoundError(p)
            file_hashes[str(p)] = sha256(p)
        wav_rows = read_scp(paths["wav"])
        spk_rows = read_scp(paths["spk1"])
        noise_rows = read_scp(paths["noise1"])
        spk_map = dict(spk_rows)
        noise_map = dict(noise_rows)
        rows: list[dict] = []
        for idx, (utt, wav_ref) in enumerate(wav_rows):
            wav_path = resolve_path(paths["wav"], wav_ref)
            fileid = parse_fileid(utt, wav_path)
            clean_path = args.dataset_root / "clean" / f"clean_fileid_{fileid}.wav" if fileid else Path("")
            noise_ref = noise_map.get(utt)
            noise_path = resolve_path(paths["noise1"], noise_ref) if noise_ref else None
            # spk1.scp is a clean path manifest in this dataset; the loader uses
            # clean_dir+fileid by default, so retain both for audit purposes.
            spk_ref = spk_map.get(utt)
            spk_path = resolve_path(paths["spk1"], spk_ref) if spk_ref else None
            rec = {
                "index": idx,
                "utt": utt,
                "wav_path": str(wav_path),
                "clean_path_loader": str(clean_path),
                "spk1_path": str(spk_path) if spk_path else None,
                "noise1_path": str(noise_path) if noise_path else None,
                "fileid": fileid,
                "speaker": parse_speaker(utt, wav_path),
                "nominal_snr_db": parse_snr(wav_path),
                "priority_bin": priority_bin(parse_snr(wav_path)),
                "filename_noise_tokens": filename_noise_tokens(wav_path),
                "noise_file_basename": noise_path.name if noise_path else None,
            }
            for key, p in (("wav", wav_path), ("clean_loader", clean_path), ("spk1", spk_path), ("noise1", noise_path)):
                if p is None:
                    rec[f"{key}_exists"] = False
                    continue
                exists = p.is_file()
                rec[f"{key}_exists"] = exists
                if not exists:
                    global_failures.append({"split": split, "utt": utt, "kind": key, "path": str(p)})
                if exists:
                    try:
                        info = sf.info(str(p))
                        rec[f"{key}_samplerate"] = int(info.samplerate)
                        rec[f"{key}_frames"] = int(info.frames)
                        rec[f"{key}_duration_s"] = float(info.frames / info.samplerate)
                        rec[f"{key}_channels"] = int(info.channels)
                    except Exception as exc:
                        rec[f"{key}_header_error"] = repr(exc)
                        global_failures.append({"split": split, "utt": utt, "kind": f"{key}_header", "path": str(p), "error": repr(exc)})
            rows.append(rec)
        records_by_split[split] = rows
        (out / f"{split}_records.jsonl").write_text(
            "".join(json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n" for r in rows), encoding="utf-8"
        )

    def aggregate(rows: list[dict]) -> dict:
        by_snr = defaultdict(list)
        by_bin = defaultdict(list)
        by_speaker = defaultdict(list)
        by_token = defaultdict(list)
        noise_files = set()
        for r in rows:
            d = r.get("wav_duration_s")
            if d is not None:
                by_snr[str(r["nominal_snr_db"])].append(d)
                by_bin[r["priority_bin"]].append(d)
                by_speaker[r["speaker"]].append(d)
                for token in r["filename_noise_tokens"]:
                    by_token[token].append(d)
            if r.get("noise_file_basename"):
                noise_files.add(r["noise_file_basename"])
        return {
            "utterances": len(rows),
            "duration_s": summary_numbers([r["wav_duration_s"] for r in rows if r.get("wav_duration_s") is not None]),
            "nominal_snr_db": {
                k: {"utterances": len(v), "duration_s": float(sum(v))}
                for k, v in sorted(by_snr.items(), key=lambda kv: (float(kv[0]) if kv[0] != "None" else math.inf))
            },
            "priority_bins": {
                k: {"utterances": len(v), "duration_s": float(sum(v))}
                for k, v in sorted(by_bin.items())
            },
            "speakers": {
                "unique": len(by_speaker),
                "per_speaker_utterances": {k: len(v) for k, v in sorted(by_speaker.items())},
                "per_speaker_duration_s": {k: float(sum(v)) for k, v in sorted(by_speaker.items())},
            },
            "filename_noise_tokens": {
                k: {"utterances": len(v), "duration_s": float(sum(v))}
                for k, v in sorted(by_token.items())
            },
            "noise_file_basename_unique": len(noise_files),
        }

    aggregate_by_split = {split: aggregate(rows) for split, rows in records_by_split.items()}
    all_rows = [r for rows in records_by_split.values() for r in rows]
    aggregate_all = aggregate(all_rows)
    tr_rows = records_by_split["tr_synthetic"]
    cv_rows = records_by_split["cv_synthetic"]
    def values(rows: list[dict], key: str) -> set:
        return {r[key] for r in rows if r.get(key) is not None}
    tr_speakers, cv_speakers = values(tr_rows, "speaker"), values(cv_rows, "speaker")
    tr_fileids, cv_fileids = values(tr_rows, "fileid"), values(cv_rows, "fileid")
    tr_noise, cv_noise = values(tr_rows, "noise_file_basename"), values(cv_rows, "noise_file_basename")
    split_intersections = {
        "speaker_overlap": sorted(tr_speakers & cv_speakers),
        "speaker_overlap_count": len(tr_speakers & cv_speakers),
        "fileid_overlap_count": len(tr_fileids & cv_fileids),
        "noise_file_basename_overlap_count": len(tr_noise & cv_noise),
        "utt_overlap_count": len(values(tr_rows, "utt") & values(cv_rows, "utt")),
    }
    consistency = {}
    for split, rows in records_by_split.items():
        utts = [r["utt"] for r in rows]
        consistency[split] = {
            "duplicate_utt": len(utts) - len(set(utts)),
            "duplicate_fileid": len([r["fileid"] for r in rows if r.get("fileid")]) - len(set(r["fileid"] for r in rows if r.get("fileid"))),
            "missing_wav": sum(not r.get("wav_exists", False) for r in rows),
            "missing_clean_loader": sum(not r.get("clean_loader_exists", False) for r in rows),
            "missing_spk1": sum(not r.get("spk1_exists", False) for r in rows),
            "missing_noise1": sum(not r.get("noise1_exists", False) for r in rows),
            "sample_rates": {key: dict(Counter(r.get(f"{key}_samplerate") for r in rows if r.get(f"{key}_samplerate") is not None)) for key in ("wav", "clean_loader", "spk1", "noise1")},
            "duration_mismatch_wav_clean": sum(abs(r.get("wav_duration_s", 0) - r.get("clean_loader_duration_s", 0)) > 1e-6 for r in rows if r.get("wav_duration_s") is not None and r.get("clean_loader_duration_s") is not None),
            "duration_mismatch_wav_noise": sum(abs(r.get("wav_duration_s", 0) - r.get("noise1_duration_s", 0)) > 1e-6 for r in rows if r.get("wav_duration_s") is not None and r.get("noise1_duration_s") is not None),
            "all_nominal_snr_parseable": all(r.get("nominal_snr_db") is not None for r in rows),
            "all_speaker_parseable": all(r.get("speaker") is not None for r in rows),
        }

    metrics = {
        "experiment_id": "p8_training_coverage_audit_20261020",
        "read_only": True,
        "dataset_root": str(args.dataset_root),
        "manifest_root": str(args.manifest_root),
        "splits_read": ["tr_synthetic", "cv_synthetic"],
        "test_manifest_read": False,
        "actual_snr_db": {"status": "unknown", "reason": "This audit reports filename nominal SNR and does not estimate power SNR for every file."},
        "file_hashes": file_hashes,
        "git": git_info(args.repo),
        "by_split": aggregate_by_split,
        "all_train_cv": aggregate_all,
        "split_intersections": split_intersections,
        "consistency": consistency,
        "failures": global_failures,
        "source_notes": {
            "noise_file_basename": "noise1.scp basenames are noise_fileid_<id>.wav and are identifiers, not semantic noise categories.",
            "filename_noise_tokens": "Tokens are weak labels parsed from noisy filenames; mixed/unknown names remain uncertain and are not ground truth categories.",
            "clean_mapping": "The observed loader defaults to clean_dir + clean_fileid_<fileid>.wav. spk1.scp is only used when train_clean_scp/valid_clean_scp are explicitly passed.",
            "noise_mapping": "noise1.scp is not consumed by WavePairsScp.__getitem__; no online noise mixing or SNR resampling is implemented there.",
        },
    }
    (out / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"out": str(out), "by_split": aggregate_by_split, "failures": len(global_failures)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
