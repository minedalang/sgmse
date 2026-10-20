#!/usr/bin/env python3
"""P8-Q-A short-time acoustic probe.

This is a clean-aware, read-only diagnostic over the existing 8-utterance
export.  It does not use transcripts or phoneme labels and does not run a
model.  For the existing 0.5 s Y_pre-suppression windows, it asks whether
10/20 ms frames contain relatively stronger clean-vs-residual energy and
whether Y_pre is low in those same frames.
"""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import numpy as np
import soundfile as sf


ROOT = Path(__file__).resolve().parent
P8 = ROOT.parent / "p8_frontend_backend_audit_20261020"
LOCAL = ROOT.parent / "p8_local_content_probe_20261020"
MANIFEST = P8 / "manifest.csv"
CANDIDATE_CSV = LOCAL / "oracle_snr_windows.csv"
FS = 16000
CENTER_START = 9.0
CENTER_SECONDS = 12.0
EPS = 1e-10
SUPPRESSION_DB = -8.0
PRIMARY_GLIMPSE_DB = -10.0
STRONG_GLIMPSE_DB = -5.0
ZERO_GLIMPSE_DB = 0.0


def read_center(path: str | Path) -> np.ndarray:
    x, fs = sf.read(str(path), always_2d=False)
    if fs != FS:
        raise ValueError(f"{path}: expected {FS} Hz, got {fs}")
    if x.ndim > 1:
        x = x.mean(axis=1)
    lo = int(round(CENTER_START * FS))
    hi = lo + int(round(CENTER_SECONDS * FS))
    if len(x) < hi:
        raise ValueError(f"{path}: only {len(x)} samples, need {hi}")
    return np.asarray(x[lo:hi], dtype=np.float64)


def frame_matrix(x: np.ndarray, frame_len: int, hop: int) -> np.ndarray:
    if len(x) < frame_len:
        return np.zeros((0, frame_len), dtype=np.float64)
    n = 1 + (len(x) - frame_len) // hop
    starts = hop * np.arange(n, dtype=np.int64)
    idx = starts[:, None] + np.arange(frame_len, dtype=np.int64)[None, :]
    return x[idx] * np.hanning(frame_len)[None, :]


def rms(x: np.ndarray) -> np.ndarray:
    return np.sqrt(np.mean(x * x, axis=1) + EPS)


def db_ratio(num: np.ndarray, den: np.ndarray) -> np.ndarray:
    return 20.0 * np.log10((num + EPS) / (den + EPS))


def normalized_logmag(x: np.ndarray) -> np.ndarray:
    mag = np.abs(np.fft.rfft(x, axis=1)) + EPS
    lm = np.log(mag)
    return lm / (np.linalg.norm(lm, axis=1, keepdims=True) + EPS)


def finite_stats(xs: np.ndarray) -> dict:
    xs = np.asarray(xs, dtype=np.float64)
    xs = xs[np.isfinite(xs)]
    if len(xs) == 0:
        return {"n": 0}
    return {
        "n": int(len(xs)),
        "mean": float(np.mean(xs)),
        "p10": float(np.quantile(xs, 0.10)),
        "median": float(np.median(xs)),
        "p90": float(np.quantile(xs, 0.90)),
        "max": float(np.max(xs)),
    }


def bool_count(mask: np.ndarray) -> dict:
    mask = np.asarray(mask, dtype=bool)
    return {"n": int(mask.size), "count": int(mask.sum()), "fraction": float(mask.mean()) if mask.size else float("nan")}


def aggregate_frame_rows(rows: list[dict], group: str, frame_ms: int) -> dict:
    selected = [r for r in rows if r["group"] == group and r["frame_ms"] == frame_ms]
    active = [r for r in selected if r["clean_active"]]
    high10 = [r for r in active if r["local_snr_db"] >= PRIMARY_GLIMPSE_DB]
    high5 = [r for r in active if r["local_snr_db"] >= STRONG_GLIMPSE_DB]
    high0 = [r for r in active if r["local_snr_db"] >= ZERO_GLIMPSE_DB]
    def vals(data: list[dict], key: str) -> np.ndarray:
        return np.asarray([float(r[key]) for r in data], dtype=np.float64)
    return {
        "group": group,
        "frame_ms": frame_ms,
        "all_frames": len(selected),
        "clean_active_frames": len(active),
        "local_snr_db_active": finite_stats(vals(active, "local_snr_db")),
        "glimpse_ge_minus10_db": bool_count(np.asarray([r["local_snr_db"] >= PRIMARY_GLIMPSE_DB for r in active], dtype=bool)),
        "glimpse_ge_minus5_db": bool_count(np.asarray([r["local_snr_db"] >= STRONG_GLIMPSE_DB for r in active], dtype=bool)),
        "glimpse_ge_0_db": bool_count(np.asarray([r["local_snr_db"] >= ZERO_GLIMPSE_DB for r in active], dtype=bool)),
        "y_pre_rms_vs_clean_db_active": finite_stats(vals(active, "y_pre_rms_vs_clean_db")),
        "clean_rms_dbfs_ge_minus10": finite_stats(vals(high10, "clean_rms_dbfs")),
        "clean_rms_dbfs_ge_minus5": finite_stats(vals(high5, "clean_rms_dbfs")),
        "clean_rms_dbfs_ge_0": finite_stats(vals(high0, "clean_rms_dbfs")),
        "y_pre_rms_vs_clean_db_ge_minus10": finite_stats(vals(high10, "y_pre_rms_vs_clean_db")),
        "y_pre_suppressed_ge_minus10": bool_count(np.asarray([r["y_pre_suppressed"] for r in high10], dtype=bool)),
        "y_pre_rms_vs_clean_db_ge_minus5": finite_stats(vals(high5, "y_pre_rms_vs_clean_db")),
        "y_pre_suppressed_ge_minus5": bool_count(np.asarray([r["y_pre_suppressed"] for r in high5], dtype=bool)),
        "y_pre_rms_vs_clean_db_ge_0": finite_stats(vals(high0, "y_pre_rms_vs_clean_db")),
        "y_pre_suppressed_ge_0": bool_count(np.asarray([r["y_pre_suppressed"] for r in high0], dtype=bool)),
        "noisy_spec_cos_ge_minus10": finite_stats(vals(high10, "noisy_spec_cos_clean")),
        "y_pre_spec_cos_ge_minus10": finite_stats(vals(high10, "y_pre_spec_cos_clean")),
        "high10_count": len(high10),
        "high5_count": len(high5),
        "high0_count": len(high0),
    }


def aggregate_windows(window_rows: list[dict], group: str, frame_ms: int) -> dict:
    selected = [r for r in window_rows if r["group"] == group and r["frame_ms"] == frame_ms]
    def vals(key: str) -> np.ndarray:
        return np.asarray([float(r[key]) for r in selected], dtype=np.float64)
    return {
        "group": group,
        "frame_ms": frame_ms,
        "windows": len(selected),
        "windows_with_ge_minus10_db": int(sum(r["n_ge_minus10"] > 0 for r in selected)),
        "windows_with_ge_minus5_db": int(sum(r["n_ge_minus5"] > 0 for r in selected)),
        "windows_with_ge_0_db": int(sum(r["n_ge_0"] > 0 for r in selected)),
        "frames_ge_minus10_db": int(sum(r["n_ge_minus10"] for r in selected)),
        "frames_ge_minus5_db": int(sum(r["n_ge_minus5"] for r in selected)),
        "frames_ge_0_db": int(sum(r["n_ge_0"] for r in selected)),
        "window_local_snr_p90_db": finite_stats(vals("local_snr_p90_db")),
        "window_local_snr_max_db": finite_stats(vals("local_snr_max_db")),
        "high10_y_pre_rms_vs_clean_db": finite_stats(vals("high10_y_pre_rms_median_db")),
        "high10_y_pre_suppressed_fraction": finite_stats(vals("high10_y_pre_suppressed_fraction")),
    }


def main() -> None:
    manifest = list(csv.DictReader(MANIFEST.open()))
    by_fileid = {str(r["fileid"]): (i, r) for i, r in enumerate(manifest)}
    candidate_rows = list(csv.DictReader(CANDIDATE_CSV.open()))
    candidates = [r for r in candidate_rows if r["candidate_frontend_suppression"].lower() == "true"]
    candidate_keys = {(str(r["fileid"]), round(float(r["start_s"]), 3), round(float(r["end_s"]), 3)) for r in candidates}
    all_keys = {(str(r["fileid"]), round(float(r["start_s"]), 3), round(float(r["end_s"]), 3)) for r in candidate_rows}

    # Load each 12 s signal once; no model call is made.
    signals: dict[str, dict[str, np.ndarray]] = {}
    for fileid, (i, rec) in by_fileid.items():
        base = Path(rec["noisy_basename"]).name
        signals[fileid] = {
            "noisy": read_center(rec["noisy_path"]),
            "clean": read_center(rec["clean_path"]),
            "y_pre": read_center(P8 / "outputs" / "y_pre" / f"{i:02d}_{base}"),
        }

    frame_rows: list[dict] = []
    window_rows: list[dict] = []
    for frame_ms in (10, 20):
        frame_len = int(round(FS * frame_ms / 1000.0))
        hop = int(round(FS * 0.010))
        for fileid, sig in signals.items():
            mats = {k: frame_matrix(v, frame_len, hop) for k, v in sig.items()}
            n = min(len(v) for v in mats.values())
            mats = {k: v[:n] for k, v in mats.items()}
            clean_rms = rms(mats["clean"])
            active_threshold = max(float(np.quantile(clean_rms, 0.30)), float(np.max(clean_rms) * 0.02))
            active = clean_rms >= active_threshold
            noisy_rms = rms(mats["noisy"])
            ypre_rms = rms(mats["y_pre"])
            noise = mats["noisy"] - mats["clean"]
            noise_rms = rms(noise)
            local_snr = db_ratio(clean_rms, noise_rms)
            noisy_spec = normalized_logmag(mats["noisy"])
            clean_spec = normalized_logmag(mats["clean"])
            ypre_spec = normalized_logmag(mats["y_pre"])
            noisy_cos = np.sum(noisy_spec * clean_spec, axis=1)
            ypre_cos = np.sum(ypre_spec * clean_spec, axis=1)
            ypre_ratio = db_ratio(ypre_rms, clean_rms)
            frame_start = np.arange(n, dtype=np.float64) * hop / FS
            for key in sorted(all_keys):
                if key[0] != fileid:
                    continue
                start_rel = key[1] - CENTER_START
                end_rel = key[2] - CENTER_START
                in_window = (frame_start >= start_rel - 1e-9) & (frame_start + frame_ms / 1000.0 <= end_rel + 1e-9)
                group = "suppression" if key in candidate_keys else "other"
                idxs = np.flatnonzero(in_window)
                # Only clean-active frames enter the local SNR and deletion summaries.
                act_idxs = idxs[active[idxs]] if len(idxs) else np.asarray([], dtype=int)
                snr_act = local_snr[act_idxs]
                ypre_act = ypre_ratio[act_idxs]
                high10 = snr_act >= PRIMARY_GLIMPSE_DB
                high5 = snr_act >= STRONG_GLIMPSE_DB
                high0 = snr_act >= ZERO_GLIMPSE_DB
                ypre_high10 = ypre_act[high10]
                row = {
                    "fileid": fileid,
                    "speaker": by_fileid[fileid][1]["speaker"],
                    "start_s": key[1],
                    "end_s": key[2],
                    "group": group,
                    "frame_ms": frame_ms,
                    "frames_total": int(len(idxs)),
                    "clean_active_frames": int(len(act_idxs)),
                    "local_snr_median_db": float(np.median(snr_act)) if len(snr_act) else float("nan"),
                    "local_snr_p90_db": float(np.quantile(snr_act, 0.90)) if len(snr_act) else float("nan"),
                    "local_snr_max_db": float(np.max(snr_act)) if len(snr_act) else float("nan"),
                    "n_ge_minus10": int(high10.sum()),
                    "n_ge_minus5": int(high5.sum()),
                    "n_ge_0": int(high0.sum()),
                    "high10_y_pre_rms_median_db": float(np.median(ypre_high10)) if len(ypre_high10) else float("nan"),
                    "high10_y_pre_suppressed_fraction": float(np.mean(ypre_high10 <= SUPPRESSION_DB)) if len(ypre_high10) else float("nan"),
                }
                window_rows.append(row)
                for j in idxs:
                    frame_rows.append({
                        "fileid": fileid,
                        "speaker": by_fileid[fileid][1]["speaker"],
                        "window_start_s": key[1],
                        "window_end_s": key[2],
                        "group": group,
                        "frame_ms": frame_ms,
                        "frame_start_s": round(CENTER_START + float(frame_start[j]), 6),
                        "clean_active": bool(active[j]),
                        "local_snr_db": float(local_snr[j]),
                        "clean_rms_dbfs": float(20.0 * np.log10(clean_rms[j] + EPS)),
                        "y_pre_rms_vs_clean_db": float(ypre_ratio[j]),
                        "y_pre_suppressed": bool(ypre_ratio[j] <= SUPPRESSION_DB),
                        "noisy_spec_cos_clean": float(noisy_cos[j]),
                        "y_pre_spec_cos_clean": float(ypre_cos[j]),
                        "noisy_rms_vs_clean_db": float(db_ratio(noisy_rms[j:j+1], clean_rms[j:j+1])[0]),
                    })

    out_csv = ROOT / "shorttime_frames.csv"
    with out_csv.open("w", newline="") as f:
        fields = list(frame_rows[0])
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(frame_rows)
    out_windows = ROOT / "shorttime_window_summary.csv"
    with out_windows.open("w", newline="") as f:
        fields = list(window_rows[0])
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(window_rows)

    summary = {
        "experiment_id": "p8_q_a_shorttime_probe_20261020",
        "protocol": {
            "source_manifest": str(MANIFEST),
            "source_windows": str(CANDIDATE_CSV),
            "utterances": len(manifest),
            "all_clean_active_windows": len(all_keys),
            "suppression_windows": len(candidate_keys),
            "other_windows": len(all_keys - candidate_keys),
            "frame_durations_ms": [10, 20],
            "frame_hop_ms": 10,
            "local_snr_definition": "20*log10(RMS(clean_frame)/RMS((noisy-clean)_frame)); clean-aware diagnostic only",
            "clean_active_definition": "frame clean RMS >= max(per-utterance 30th percentile, 2% of peak)",
            "primary_glimpse_threshold_db": PRIMARY_GLIMPSE_DB,
            "strong_glimpse_threshold_db": STRONG_GLIMPSE_DB,
            "zero_glimpse_threshold_db": ZERO_GLIMPSE_DB,
            "y_pre_suppression_threshold_db": SUPPRESSION_DB,
            "spectral_feature": "cosine of normalized log magnitude spectrum; acoustic proxy only",
            "no_transcript_or_phoneme_labels": True,
            "clean_reference_source": "paired clean WAV in DNS-Ins20 synthetic manifest; no verified transcript used",
            "model_inference": False,
            "dataset_modified": False,
        },
        "by_frame": [aggregate_frame_rows(frame_rows, group, frame_ms) for frame_ms in (10, 20) for group in ("suppression", "other")],
        "by_window": [aggregate_windows(window_rows, group, frame_ms) for frame_ms in (10, 20) for group in ("suppression", "other")],
    }
    (ROOT / "metrics.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
