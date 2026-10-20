#!/usr/bin/env python3
"""Corrected integer-frame fixed-time probe; refuses existing result artifacts."""

from __future__ import annotations

import csv
import hashlib
import json
from datetime import datetime, timezone
import io
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
NEIGHBORHOOD_FRAMES = 10
GROUP_RE = re.compile(r"(?P<group>\d+_fileid_\d+_[^_]+-[^_]+s)_context_(?P<condition>clean|noisy|y_pre|off|router)\.wav$")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    output_names = ("fixed_time_token_posteriors.csv", "frame_probabilities.npz", "metrics.json", "run_started.json")
    existing = [name for name in output_names if (ROOT / name).exists()]
    if existing:
        raise FileExistsError(f"Refusing to overwrite result artifacts: {existing}")
    started_at = datetime.now(timezone.utc).isoformat()
    with (ROOT / "run_started.json").open("x") as handle:
        json.dump({"started_at_utc": started_at, "status": "started"}, handle)
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
    cached_probabilities = {}
    frame_cache_manifest = []
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
            cache_key = f"clip_{len(frame_cache_manifest):02d}"
            target_ids = np.array(sorted({int(t["token_id"]) for t in tokens}), dtype=np.int32)
            cached_probabilities[cache_key + "_probabilities"] = probs[:, target_ids]
            cached_probabilities[cache_key + "_token_ids"] = target_ids
            cached_probabilities[cache_key + "_argmax_ids"] = probs.argmax(axis=-1).astype(np.int32)
            frame_cache_manifest.append({"array_key": cache_key, "group": group, "condition": condition,
                                         "path": str(path), "frames": len(probs), "vocabulary_size": probs.shape[1],
                                         "cached_token_ids": target_ids.tolist(),
                                         "probabilities_key": cache_key + "_probabilities",
                                         "token_ids_key": cache_key + "_token_ids",
                                         "argmax_ids_key": cache_key + "_argmax_ids"})
            ids = probs.argmax(axis=-1)
            for token in tokens:
                start = int(token["clean_greedy_start_frame"])
                end = int(token["clean_greedy_end_frame_exclusive"])
                token_id = int(token["token_id"])
                if not 0 <= start < end <= len(probs):
                    raise ValueError(f"Invalid source interval {group}/{condition}: [{start}, {end}) for {len(probs)} frames")
                lo, hi = start, end
                fixed = probs[lo:hi, token_id]
                nlo = max(0, start - NEIGHBORHOOD_FRAMES)
                nhi = min(len(probs), end + NEIGHBORHOOD_FRAMES)
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

    cache_buffer = io.BytesIO()
    np.savez_compressed(cache_buffer, **cached_probabilities)
    cache_payload = cache_buffer.getvalue()
    if len(cache_payload) >= 1_000_000:
        raise ValueError(f"Probability cache exceeds 1 MB: {len(cache_payload)} bytes")
    cache_path = ROOT / "frame_probabilities.npz"
    with cache_path.open("xb") as handle:
        handle.write(cache_payload)

    out_csv = ROOT / "fixed_time_token_posteriors.csv"
    with out_csv.open("x", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)

    summary = {
        "experiment_id": "p8_q_a_ctc_fixed_time_v2_20261018",
        "started_at_utc": started_at,
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "seed": 0,
        "source_metrics_sha256": sha256(SOURCE_METRICS),
        "probe_code_sha256": sha256(Path(__file__)),
        "config_sha256": sha256(ROOT / "config.yaml"),
        "token_csv_sha256": sha256(out_csv),
        "probability_cache": {"path": str(cache_path), "sha256": sha256(cache_path),
                              "bytes": len(cache_payload), "dtype": "float32",
                              "content": "All frames of clean-target token probabilities, token-ID column map, and full-vocabulary frame argmax IDs. Float32 target values preserve exact metric recomputation; logits and other class probabilities not cached.",
                              "manifest": frame_cache_manifest},
        "tokenizer_vocab_sha256": sha256(MODEL / "vocab.json"),
        "status": "completed",
        "model": "facebook/wav2vec2-lv-60-espeak-cv-ft",
        "model_weight_sha256": sha256(MODEL / "pytorch_model.bin"),
        "device": DEVICE,
        "groups": len(groups),
        "conditions_per_group": len(expected),
        "token_condition_rows": len(records),
        "fixed_window_definition": "integer [clean_greedy_start_frame, clean_greedy_end_frame_exclusive), held fixed across all conditions",
        "nearby_window_definition": "integer [max(0,start-10), min(num_frames,end+10))",
        "nearby_window_frames_each_side": NEIGHBORHOOD_FRAMES,
        "nearby_window_seconds_each_side": NEIGHBORHOOD_S,
        "frame_step_seconds_assumed_from_source_probe": FRAME_STEP_S,
        "runtime_seconds": time.time() - t0,
        "interpretation": "clean-derived CTC pseudo-target acoustic support at fixed clean timestamps and nearby interval; not verified phoneme truth, WER/CER, or human recoverability",
        "source_sha256": {str(p): sha256(p) for arms in groups.values() for p in arms.values()},
        "dataset_modified": False,
    }
    with (ROOT / "metrics.json").open("x") as handle:
        handle.write(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
