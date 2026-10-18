#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
from pathlib import Path

import soundfile as sf


ROOT = Path(__file__).resolve().parent
P8 = ROOT.parent / "p8_frontend_backend_audit_20261020"
PROBE = ROOT.parent / "p8_local_content_probe_20261020"
OUT = ROOT / "clips"
OUT.mkdir(parents=True, exist_ok=True)

# Higher-cue candidates: suppression plus local oracle SNR > -10 dB.
CANDIDATES = [
    {"index": 0, "fileid": "5243", "start_s": 19.5, "end_s": 20.0},
    {"index": 2, "fileid": "4812", "start_s": 12.5, "end_s": 13.0},
    {"index": 2, "fileid": "4812", "start_s": 15.0, "end_s": 15.5},
    {"index": 2, "fileid": "4812", "start_s": 20.5, "end_s": 20.97},
]

def read(path):
    x, fs = sf.read(str(path), always_2d=False)
    if x.ndim > 1:
        x = x.mean(axis=1)
    return x, fs


def main():
    manifest = list(csv.DictReader((P8 / "manifest.csv").open()))
    rows = list(csv.DictReader((PROBE / "oracle_snr_windows.csv").open()))
    cue_rows = list(csv.DictReader((PROBE / "noisy_cue_windows.csv").open()))
    arms = ["noisy", "clean", "y_pre", "off", "router"]
    records = []
    for j, cand in enumerate(CANDIDATES):
        i = cand["index"]
        rec = manifest[i]
        base = Path(rec["noisy_basename"]).name
        paths = {
            "noisy": Path(rec["noisy_path"]),
            "clean": Path(rec["clean_path"]),
            "y_pre": P8 / "outputs" / "y_pre" / f"{i:02d}_{base}",
            "off": P8 / "outputs" / "off" / f"{i:02d}_{base}",
            "router": P8 / "outputs" / "router" / f"{i:02d}_{base}",
        }
        a = int(round(cand["start_s"] * 16000))
        b = int(round(cand["end_s"] * 16000))
        stem = f"{j:02d}_fileid_{cand['fileid']}_{cand['start_s']:.2f}-{cand['end_s']:.2f}s"
        for arm in arms:
            x, fs = read(paths[arm])
            sf.write(str(OUT / f"{stem}_{arm}.wav"), x[a:b], fs, subtype="PCM_16")
        matched = [r for r in rows if int(r["index"]) == i and abs(float(r["start_s"]) - cand["start_s"]) < .01]
        matched_cue = [r for r in cue_rows if int(r["index"]) == i and abs(float(r["start_s"]) - cand["start_s"]) < .01]
        records.append({**cand,
                        "oracle_local_snr_db": float(matched[0]["oracle_local_snr_db"]),
                        "y_pre_rms_db_vs_clean": float(matched[0]["y_pre_rms_db_vs_clean"]),
                        "off_rms_db_vs_clean": float(matched[0]["off_rms_db_vs_clean"]),
                        "router_rms_db_vs_clean": float(matched[0]["router_rms_db_vs_clean"]),
                        "noisy_clean_wave_corr": float(matched_cue[0]["noisy_clean_wave_corr"]),
                        "clean_context": rec["fileid"]})
    (ROOT / "candidates.json").write_text(json.dumps(records, indent=2) + "\n")


if __name__ == "__main__":
    main()
