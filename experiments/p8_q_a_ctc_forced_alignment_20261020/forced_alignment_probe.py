#!/usr/bin/env python3
"""Fixed-clean-target CTC forced-alignment pilot for P8-Q-A.

The clean clip supplies a high-confidence *pseudo*-target sequence.  The
sequence is then held fixed while Viterbi CTC paths are computed for noisy,
Y_pre, Router-off, and Router clips.  This is an acoustic-model probe only;
the clean CTC sequence is not a verified transcript or phoneme annotation.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
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
MODEL = SGMSE / "experiments/p8_phoneme_probe_20261020/local_model"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 0
FS = 16000
CLEAN_RUN_POSTERIOR_THRESHOLD = 0.50
CONDS = ("clean", "noisy", "y_pre", "off", "router")
GROUP_RE = re.compile(r"(?P<group>\d+_fileid_\d+_[^_]+-[^_]+s)_context_(?P<condition>clean|noisy|y_pre|off|router)\.wav$")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def collapse_runs(ids: np.ndarray, probs: np.ndarray, blank: int) -> list[dict]:
    """Return nonblank greedy CTC runs with their frame boundaries."""
    runs: list[dict] = []
    i = 0
    while i < len(ids):
        j = i + 1
        while j < len(ids) and int(ids[j]) == int(ids[i]):
            j += 1
        token = int(ids[i])
        if token != blank:
            runs.append({
                "token_id": token,
                "start_frame": int(i),
                "end_frame_exclusive": int(j),
                "frames": int(j - i),
                "posterior_mean": float(np.mean(probs[i:j, token])),
                "posterior_min": float(np.min(probs[i:j, token])),
                "posterior_max": float(np.max(probs[i:j, token])),
            })
        i = j
    return runs


def ctc_viterbi(log_probs: np.ndarray, target: list[int], blank: int) -> tuple[np.ndarray | None, float | None]:
    """Maximum-score CTC path constrained to target, with frame states returned."""
    t_len = int(log_probs.shape[0])
    if not target:
        return np.zeros(t_len, dtype=np.int64), float(np.sum(log_probs[:, blank]))
    ext = [blank]
    for token in target:
        ext.extend([int(token), blank])
    s_len = len(ext)
    if t_len == 0 or t_len < len(target):
        return None, None
    neg_inf = -np.inf
    dp = np.full((t_len, s_len), neg_inf, dtype=np.float64)
    back = np.full((t_len, s_len), -1, dtype=np.int16)
    dp[0, 0] = log_probs[0, blank]
    dp[0, 1] = log_probs[0, target[0]]
    for t in range(1, t_len):
        for s, token in enumerate(ext):
            candidates = [(dp[t - 1, s], s)]
            if s > 0:
                candidates.append((dp[t - 1, s - 1], s - 1))
            if s > 1 and token != blank and token != ext[s - 2]:
                candidates.append((dp[t - 1, s - 2], s - 2))
            prev_score, prev_state = max(candidates, key=lambda x: x[0])
            if np.isfinite(prev_score):
                dp[t, s] = prev_score + log_probs[t, token]
                back[t, s] = prev_state
    end_candidates = [(dp[-1, s_len - 1], s_len - 1), (dp[-1, s_len - 2], s_len - 2)]
    score, state = max(end_candidates, key=lambda x: x[0])
    if not np.isfinite(score):
        return None, None
    path = np.full(t_len, -1, dtype=np.int64)
    path[-1] = state
    for t in range(t_len - 1, 0, -1):
        state = int(back[t, state])
        if state < 0:
            return None, None
        path[t - 1] = state
    # Every target state must be visited; otherwise the constrained path is
    # incomplete even if the terminal dynamic-program score is finite.
    if any(not np.any(path == (2 * i + 1)) for i in range(len(target))):
        return None, None
    return path, float(score)


def load_logits(proc, model, path: Path) -> dict:
    wav, sr = sf.read(str(path), dtype="float32")
    if sr != FS:
        raise ValueError(f"{path}: expected {FS}, got {sr}")
    if wav.ndim > 1:
        wav = wav.mean(axis=1)
    inp = proc(wav, sampling_rate=sr, return_tensors="pt")
    with torch.inference_mode():
        logits = model(inp.input_values.to(DEVICE)).logits[0].float().cpu().numpy()
    logits = logits - np.max(logits, axis=-1, keepdims=True)
    probs = np.exp(logits)
    probs /= np.sum(probs, axis=-1, keepdims=True)
    ids = probs.argmax(axis=-1)
    return {
        "path": str(path),
        "sha256": sha256(path),
        "frames": int(len(ids)),
        "probs": probs,
        "ids": ids,
    }


def token_name(proc, token_id: int) -> str:
    return str(proc.tokenizer.convert_ids_to_tokens(int(token_id)))


def main() -> None:
    torch.manual_seed(SEED)
    proc = Wav2Vec2Processor.from_pretrained(str(MODEL), local_files_only=True)
    model = Wav2Vec2ForCTC.from_pretrained(str(MODEL), local_files_only=True).eval().to(DEVICE)
    blank = int(proc.tokenizer.pad_token_id)
    groups: dict[str, dict[str, Path]] = {}
    for path in sorted(CLIPS.glob("*_context_*.wav")):
        m = GROUP_RE.match(path.name)
        if m:
            groups.setdefault(m.group("group"), {})[m.group("condition")] = path
    if len(groups) != 4 or any(set(x) != set(CONDS) for x in groups.values()):
        raise RuntimeError(f"expected 4 groups x 5 conditions, got {groups}")

    t0 = time.time()
    outputs: dict[tuple[str, str], dict] = {}
    for group in sorted(groups):
        for condition in CONDS:
            outputs[(group, condition)] = load_logits(proc, model, groups[group][condition])

    group_records: list[dict] = []
    token_records: list[dict] = []
    align_records: list[dict] = []
    failures: list[dict] = []
    for group in sorted(groups):
        clean = outputs[(group, "clean")]
        clean_runs = collapse_runs(clean["ids"], clean["probs"], blank)
        target_runs = [r for r in clean_runs if r["posterior_mean"] >= CLEAN_RUN_POSTERIOR_THRESHOLD]
        target = [int(r["token_id"]) for r in target_runs]
        target_tokens = []
        for idx, r in enumerate(target_runs):
            target_tokens.append({
                "target_index": idx,
                "token_id": int(r["token_id"]),
                "token": token_name(proc, r["token_id"]),
                "clean_greedy_start_frame": r["start_frame"],
                "clean_greedy_end_frame_exclusive": r["end_frame_exclusive"],
                "clean_greedy_start_s": float(r["start_frame"] * 0.02),
                "clean_greedy_end_s": float(r["end_frame_exclusive"] * 0.02),
                "clean_run_posterior_mean": r["posterior_mean"],
                "clean_run_posterior_min": r["posterior_min"],
                "clean_run_posterior_max": r["posterior_max"],
            })
        dropped = [r for r in clean_runs if r["posterior_mean"] < CLEAN_RUN_POSTERIOR_THRESHOLD]
        group_records.append({
            "group": group,
            "blank_id": blank,
            "clean_greedy_run_count": len(clean_runs),
            "clean_target_token_count": len(target),
            "clean_low_confidence_runs_dropped": len(dropped),
            "clean_target_tokens": target_tokens,
            "clean_dropped_runs": [{"token_id": r["token_id"], "token": token_name(proc, r["token_id"]), "posterior_mean": r["posterior_mean"]} for r in dropped],
        })
        for condition in CONDS:
            out = outputs[(group, condition)]
            probs, ids = out["probs"], out["ids"]
            path, score = ctc_viterbi(np.log(probs + 1e-12), target, blank)
            greedy_blank_fraction = float(np.mean(ids == blank))
            blank_posterior_mean = float(np.mean(probs[:, blank]))
            if path is None:
                failures.append({"group": group, "condition": condition, "target_tokens": len(target), "frames": len(ids)})
                align_records.append({"group": group, "condition": condition, "alignment_success": False, "frames": len(ids), "target_tokens": len(target), "greedy_blank_fraction": greedy_blank_fraction, "blank_posterior_mean": blank_posterior_mean})
                continue
            forced_token_mask = np.zeros(len(path), dtype=bool)
            token_posteriors = []
            for ti, target_id in enumerate(target):
                state = 2 * ti + 1
                frame_idx = np.flatnonzero(path == state)
                forced_token_mask[frame_idx] = True
                posterior = probs[frame_idx, target_id]
                forced_top1 = ids[frame_idx] == target_id
                base = target_tokens[ti]
                rec = {
                    "group": group,
                    "condition": condition,
                    "target_index": ti,
                    "token_id": int(target_id),
                    "token": token_name(proc, target_id),
                    "clean_run_posterior_mean": base["clean_run_posterior_mean"],
                    "frames": int(len(frame_idx)),
                    "start_frame": int(frame_idx[0]),
                    "end_frame_exclusive": int(frame_idx[-1] + 1),
                    "start_s": float(frame_idx[0] * 0.02),
                    "end_s": float((frame_idx[-1] + 1) * 0.02),
                    "target_token_posterior_mean": float(np.mean(posterior)),
                    "target_token_posterior_min": float(np.min(posterior)),
                    "target_token_posterior_max": float(np.max(posterior)),
                    "target_token_top1_fraction": float(np.mean(forced_top1)),
                    "forced_path_logprob_mean": float(score / len(path)),
                }
                token_records.append(rec)
                token_posteriors.extend(posterior.tolist())
            blank_path = ~forced_token_mask
            align_records.append({
                "group": group,
                "condition": condition,
                "alignment_success": True,
                "frames": len(path),
                "target_tokens": len(target),
                "greedy_blank_fraction": greedy_blank_fraction,
                "blank_posterior_mean": blank_posterior_mean,
                "forced_blank_fraction": float(np.mean(blank_path)),
                "blank_fraction_delta_vs_clean_greedy": None,
                "blank_posterior_delta_vs_clean": None,
                "forced_target_posterior_mean": float(np.mean(token_posteriors)) if token_posteriors else None,
                "forced_target_posterior_median": float(np.median(token_posteriors)) if token_posteriors else None,
                "forced_target_top1_fraction": float(np.mean(ids[forced_token_mask] == np.asarray([target[s // 2] for s in path[forced_token_mask]], dtype=np.int64))) if forced_token_mask.any() else None,
                "forced_path_logprob_mean": float(score / len(path)),
            })

        clean_align = next(x for x in align_records if x["group"] == group and x["condition"] == "clean")
        for x in align_records:
            if x["group"] == group and x.get("alignment_success"):
                x["blank_fraction_delta_vs_clean_greedy"] = float(x["greedy_blank_fraction"] - clean_align["greedy_blank_fraction"])
                x["blank_posterior_delta_vs_clean"] = float(x["blank_posterior_mean"] - clean_align["blank_posterior_mean"])

    # Add a compact summary for each group/condition and aggregate counts.
    successful = [x for x in align_records if x.get("alignment_success")]
    aggregate: dict[str, dict] = {}
    for condition in CONDS:
        rows = [x for x in successful if x["condition"] == condition]
        aggregate[condition] = {
            "groups": len(rows),
            "alignment_failures": sum(1 for x in align_records if x["condition"] == condition and not x.get("alignment_success")),
            "alignment_failure_rate": float(sum(1 for x in align_records if x["condition"] == condition and not x.get("alignment_success")) / len(groups)),
            "greedy_blank_fraction_mean": float(np.mean([x["greedy_blank_fraction"] for x in rows])) if rows else None,
            "blank_posterior_mean": float(np.mean([x["blank_posterior_mean"] for x in rows])) if rows else None,
            "forced_blank_fraction_mean": float(np.mean([x["forced_blank_fraction"] for x in rows])) if rows else None,
            "forced_target_posterior_mean": float(np.mean([x["forced_target_posterior_mean"] for x in rows])) if rows else None,
            "forced_target_top1_fraction_mean": float(np.mean([x["forced_target_top1_fraction"] for x in rows])) if rows else None,
        }

    for row in token_records:
        row.pop("forced_path_logprob_mean", None)
    metrics = {
        "experiment_id": "p8_q_a_ctc_forced_alignment_20261020",
        "status": "completed",
        "model": "facebook/wav2vec2-lv-60-espeak-cv-ft",
        "model_local_dir": str(MODEL),
        "model_weight_sha256": sha256(MODEL / "pytorch_model.bin"),
        "device": DEVICE,
        "seed": SEED,
        "blank_id": blank,
        "groups": len(groups),
        "conditions_per_group": len(CONDS),
        "clean_target_definition": "clean greedy CTC nonblank runs with mean run posterior >= 0.50; fixed per group for all conditions",
        "interpretation": "clean-derived CTC pseudo-target acoustic probe; not verified transcript, phoneme truth, WER/CER, or human content recovery",
        "clip_duration_s": 2.0,
        "runtime_seconds": time.time() - t0,
        "groups_detail": group_records,
        "input_hashes": {group: {condition: outputs[(group, condition)]["sha256"] for condition in CONDS} for group in sorted(groups)},
        "alignments": align_records,
        "token_alignments": token_records,
        "aggregate": aggregate,
        "alignment_failures": failures,
    }
    (ROOT / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2) + "\n")
    with (ROOT / "token_alignments.csv").open("w", newline="") as f:
        fields = list(token_records[0]) if token_records else []
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(token_records)
    with (ROOT / "alignments.csv").open("w", newline="") as f:
        fields = list(align_records[0])
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(align_records)
    print(json.dumps({"groups": len(groups), "alignment_failures": failures, "aggregate": aggregate}, indent=2))


if __name__ == "__main__":
    main()
