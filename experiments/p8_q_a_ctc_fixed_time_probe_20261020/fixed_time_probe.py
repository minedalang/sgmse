#!/usr/bin/env python3
"""Probe clean-derived CTC tokens at fixed clean timestamps across conditions."""

from __future__ import annotations

import csv
import hashlib
import json
import re
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor


ROOT = Path(__file__).resolve().parent
SGMSE = ROOT.parents[1]
CLIPS = SGMSE / "experiments/p8_e1_candidate_listening_aligned_20261020/clips"
SOURCE_METRICS = SGMSE / "experiments/p8_q_a_ctc_forced_alignment_20261020/metrics.json"
MODEL = SGMSE / "experiments/p8_phoneme_probe_20261020/local_model"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
FS = 16000
FRAME_STEP_S = 0.02
NEIGHBORHOOD_S = 0.20
GROUP_RE = re.compile(r"(?P<group>\d+_fileid_\d+_[^_]+-[^_]+s)_context_(?P<condition>clean|noisy|y_pre|off|router)\.wav$")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    torch.manual_seed(0)
    proc = Wav2Vec2Processor.from_pretrained(str(MODEL), local_files_only=True)
    model = Wav2Vec2ForCTC.from_pretrained(str(MODEL), local_files_only=True).eval().to(DEVICE)
    groups: dict[str, dict[str, Path]] = {}
    for path in sorted(CLIPS.glob("*_context_*.wav")):
        match = GROUP_RE.match(path.name)
        if match:
            groups.setdefault(match.group("group"), {})[match.group("condition")] = path
    expected = {"clean", "noisy", "y_pre", "off", "router"}
    if len(groups) != 4 or any(set(arms) != expected for arms in groups.values()):
        raise RuntimeError(f"expected 4 groups x 5 arms, found {groups}")

    source = json.loads(SOURCE_METRICS.read_text())
    records: list[dict] = []
    t0 = time.time()
    for group in sorted(groups):
        tokens = next(x["clean_target_tokens"] for x in source["groups_detail"] if x["group"] == group)
        for condition, path in sorted(groups[group].items()):
            wav, sr = sf.read(str(path), dtype="float32")
            if sr != FS:
                raise ValueError(f"{path}: expected {FS} Hz, got {sr}")
            if wav.ndim > 1:
                wav = wav.mean(axis=1)
            inp = proc(wav, sampling_rate=sr, return_tensors="pt")
            with torch.inference_mode():
                logits = model(inp.input_values.to(DEVICE)).logits[0].float()
                probs = torch.softmax(logits, dim=-1).cpu().numpy()
            ids = probs.argmax(axis=-1)
            for token in tokens:
                start = int(token["clean_greedy_start_frame"])
                end = int(token["clean_greedy_end_frame_exclusive"])
                token_id = int(token["token_id"])
                lo = max(0, int(np.floor(token["clean_greedy_start_s"] / FRAME_STEP_S)))
                hi = min(len(probs), max(lo + 1, int(np.ceil(token["clean_greedy_end_s"] / FRAME_STEP_S))))
                fixed = probs[lo:hi, token_id]
                nlo = max(0, int(np.floor((token["clean_greedy_start_s"] - NEIGHBORHOOD_S) / FRAME_STEP_S)))
                nhi = min(len(probs), int(np.ceil((token["clean_greedy_end_s"] + NEIGHBORHOOD_S) / FRAME_STEP_S)))
                neighborhood = probs[nlo:nhi, token_id]
                records.append({
                    "group": group,
                    "condition": condition,
                    "target_index": int(token["target_index"]),
                    "token": token["token"],
                    "token_id": token_id,
                    "clean_pseudo_posterior": float(token["clean_run_posterior_mean"]),
                    "clean_start_s": float(token["clean_greedy_start_s"]),
                    "clean_end_s": float(token["clean_greedy_end_s"]),
                    "clean_start_frame": start,
                    "clean_end_frame_exclusive": end,
                    "fixed_window_frames": int(len(fixed)),
                    "fixed_time_target_posterior_mean": float(np.mean(fixed)),
                    "fixed_time_target_top1_fraction": float(np.mean(ids[lo:hi] == token_id)),
                    "nearby_window_s": NEIGHBORHOOD_S,
                    "nearby_target_posterior_max": float(np.max(neighborhood)),
                    "nearby_target_posterior_mean": float(np.mean(neighborhood)),
                    "nearby_target_top1_fraction": float(np.mean(ids[nlo:nhi] == token_id)),
                })

    out_csv = ROOT / "fixed_time_token_posteriors.csv"
    with out_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)

    summary = {
        "experiment_id": "p8_q_a_ctc_fixed_time_probe_20261020",
        "status": "completed",
        "model": "facebook/wav2vec2-lv-60-espeak-cv-ft",
        "model_weight_sha256": sha256(MODEL / "pytorch_model.bin"),
        "device": DEVICE,
        "groups": len(groups),
        "conditions_per_group": len(expected),
        "token_condition_rows": len(records),
        "fixed_window_definition": "clean CTC greedy token frame interval, held fixed across all conditions",
        "nearby_window_seconds_each_side": NEIGHBORHOOD_S,
        "frame_step_seconds_assumed_from_source_probe": FRAME_STEP_S,
        "runtime_seconds": time.time() - t0,
        "interpretation": "clean-derived CTC pseudo-target acoustic support at fixed clean timestamps and nearby interval; not verified phoneme truth, WER/CER, or human recoverability",
        "source_sha256": {str(p): sha256(p) for arms in groups.values() for p in arms.values()},
        "dataset_modified": False,
    }
    (ROOT / "metrics.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
