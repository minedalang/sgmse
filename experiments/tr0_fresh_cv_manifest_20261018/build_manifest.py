#!/usr/bin/env python3
"""Build a deterministic fresh CV manifest for the TR-0 adapter screen.

This script only reads the DNS Ins20 SCP metadata and writes a small manifest;
it never copies, rewrites, or modifies dataset audio.
"""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path


ROOT = Path("/data/coding/sgmse")
DATA = Path("/data/datasets/dns_ins20")
OUT = ROOT / "experiments/tr0_fresh_cv_manifest_20261018"
SOURCE = DATA / "dns_snr_-20_-5/synthetic_snr-20_-5/scp_new/cv_synthetic"
OLD32 = ROOT / "experiments/rrsd_cv_repeat32/manifest.csv"
OLD64 = ROOT / "experiments/speaker_headroom_replication_64_20261017/manifest.csv"
EXCLUDE = {"reader_10658"}


def read_speakers(path: Path, speaker_col: int) -> set[str]:
    speakers: set[str] = set()
    with path.open(newline="") as f:
        for row in csv.reader(f):
            if not row or row[0].startswith("filename"):
                continue
            speakers.add(row[speaker_col])
    return speakers


def read_scp(path: Path) -> dict[str, str]:
    out: dict[str, str] = {}
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            key, value = line.split(None, 1)
            out[key] = value
    return out


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    excluded = set(EXCLUDE)
    excluded |= read_speakers(OLD32, 2)
    excluded |= read_speakers(OLD64, 2)

    wav = read_scp(SOURCE / "wav.scp")
    clean = read_scp(SOURCE / "spk1.scp")
    utt2spk = read_scp(SOURCE / "utt2spk")
    if set(wav) != set(clean) or set(wav) != set(utt2spk):
        raise RuntimeError("SCP key sets do not match")

    by_speaker: dict[str, list[str]] = {}
    for uttid in sorted(wav):
        speaker = utt2spk[uttid]
        if speaker in excluded:
            continue
        by_speaker.setdefault(speaker, []).append(uttid)
    eligible = sorted(s for s, utts in by_speaker.items() if len(utts) >= 2)
    selected = eligible[:32]
    if len(selected) != 32:
        raise RuntimeError(f"only {len(selected)} speakers have >=2 utterances")

    rows = []
    for idx, speaker in enumerate(selected):
        for uttid in by_speaker[speaker][:2]:
            fileid = uttid.rsplit("_fileid_", 1)[-1]
            noisy_rel = wav[uttid]
            clean_rel = clean[uttid]
            noisy = (SOURCE / noisy_rel).resolve()
            clean_path = (SOURCE / clean_rel).resolve()
            if not noisy.exists() or not clean_path.exists():
                raise FileNotFoundError(f"missing pair for {uttid}: {noisy} / {clean_path}")
            rows.append(
                {
                    "filename": noisy.name,
                    "uttid": uttid,
                    "speaker": speaker,
                    "fileid": fileid,
                    "noisy": str(noisy),
                    "clean": str(clean_path),
                    "source_split": "dns_snr_-20_-5/scp_new/cv_synthetic",
                    "selection_index": idx,
                }
            )

    manifest = OUT / "manifest.csv"
    fields = list(rows[0])
    with manifest.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    lock = {
        "manifest_sha256": sha256_file(manifest),
        "source_metadata": {
            "source_dir": str(SOURCE),
            "source_files": {name: sha256_file(SOURCE / name) for name in ["wav.scp", "spk1.scp", "utt2spk"]},
        },
        "selection_rule": "sort eligible speaker IDs; exclude old 16-speaker rrsd_cv_repeat32, old 64-speaker speaker_headroom_replication_64_20261017, and reader_10658; take first 32 speakers and first two sorted uttids each",
        "excluded_speakers": sorted(excluded),
        "selected_speakers": selected,
        "n_speakers": len(selected),
        "n_utterances": len(rows),
        "dataset_modified": False,
    }
    (OUT / "manifest_lock.json").write_text(json.dumps(lock, indent=2) + "\n")
    (OUT / "manifest.sha256").write_text(f"{lock['manifest_sha256']}  manifest.csv\n")
    print(json.dumps({k: lock[k] for k in ["manifest_sha256", "n_speakers", "n_utterances", "selected_speakers"]}, indent=2))


if __name__ == "__main__":
    main()
