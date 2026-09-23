#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Oracle Complementarity Upper-Bound experiment for this SGMSE joint project.

Purpose
-------
Use ONE trained no-selector JointSemambaSgmseModel checkpoint and the clean test
reference to measure how much complementary information exists between:
  1) the embedded frontend enhancer output, and
  2) the diffusion backend output.

This is an oracle diagnostic only. Clean speech is deliberately used to choose
which model is better, so the oracle outputs are NOT deployable systems.

Outputs
-------
<out_dir>/frontend/*.wav
<out_dir>/backend/*.wav
<out_dir>/oracle_utt/*.wav
<out_dir>/oracle_patch/*.wav
<out_dir>/oracle_projection/*.wav
<out_dir>/per_file_metrics.csv
<out_dir>/summary.csv
<out_dir>/oracle_stats.csv
<out_dir>/RUN_INFO.txt

Oracle definitions
------------------
A) Utterance oracle:
   Choose frontend or backend for the whole utterance according to global
   compressed-complex-spectrum MSE to clean.

B) Patch hard oracle:
   In the same compressed complex spectral domain used by the model, compute
   local frontend/backend errors with stride-1 AvgPool (default 5x3, matching
   frontend_nr_patch_freq/time). Use frontend wherever its local error is lower.

C) Patch projection oracle:
   Let B = backend spectrum, F = frontend spectrum, X = clean spectrum,
       D = F - B, R = X - B.
   Solve a local least-squares interpolation coefficient
       alpha* = clip( <D,R> / (||D||^2 + eps), 0, 1 )
   using the same local patch averaging, and output
       X_oracle = B + alpha* (F - B).
   This is a stronger upper bound than hard selection because it can use an
   optimal convex interpolation between frontend and backend.

Example
-------
CUDA_VISIBLE_DEVICES=0 python oracle_complementarity_analysis.py \
  --ckpt /path/to/no_selector.ckpt \
  --noisy_dir /path/to/test/noisy \
  --clean_dir /path/to/test/clean \
  --out_dir oracle_no_selector \
  --use_ema --N 30 \
  --patch_freq 5 --patch_time 3
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import librosa
import numpy as np
import pandas as pd
import soundfile as sf
import torch
import torch.nn.functional as F

from pesq import pesq
from pystoi import stoi

from sgmse.joint_model import JointSemambaSgmseModel
from sgmse.joint_data_module import JointWaveDataModule
from sgmse.util.other import energy_ratios


# -----------------------------------------------------------------------------
# I/O helpers
# -----------------------------------------------------------------------------

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_mono(path: str, target_sr: int) -> np.ndarray:
    wav, sr = sf.read(path, always_2d=False)
    if wav.ndim > 1:
        wav = wav.mean(axis=-1)
    wav = wav.astype(np.float32, copy=False)
    if int(sr) != int(target_sr):
        wav = librosa.resample(wav, orig_sr=int(sr), target_sr=int(target_sr)).astype(np.float32)
    return wav


def match_np_length(x: np.ndarray, target_len: int) -> np.ndarray:
    """Center crop/pad, matching the project inference convention."""
    n = int(x.shape[-1])
    if n == target_len:
        return x
    if n > target_len:
        start = (n - target_len) // 2
        return x[start:start + target_len]
    pad = target_len - n
    left = pad // 2
    right = pad - left
    return np.pad(x, (left, right), mode="constant")


def match_torch_length(y: torch.Tensor, target_len: int) -> torch.Tensor:
    cur = int(y.size(-1))
    if cur == target_len:
        return y
    if cur > target_len:
        start = (cur - target_len) // 2
        return y[..., start:start + target_len]
    pad = target_len - cur
    left = pad // 2
    right = pad - left
    return F.pad(y, (left, right), mode="constant", value=0.0)


def list_wavs(root: str) -> List[Path]:
    root_p = Path(root)
    return sorted(p for p in root_p.rglob("*.wav") if p.is_file())


def resolve_clean_path(noisy_path: Path, noisy_root: Path, clean_root: Path, pairing: str) -> Path:
    """Resolve a noisy file to its clean reference.

    Supported schemes:
      * exact: identical relative path / basename
      * dns: DNS-style fallbacks, including this test-set convention:
            <long_noisy_name>_fileid_56.wav -> clean_fileid_56.wav
      * auto: try exact first, then all DNS fallbacks
    """
    rel = noisy_path.relative_to(noisy_root)
    basename = noisy_path.name
    stem = noisy_path.stem

    candidates: List[Path] = []

    if pairing in ("auto", "exact"):
        candidates += [clean_root / rel, clean_root / basename]

    if pairing in ("auto", "dns"):
        # DNS synthetic test-set convention used by the user's data:
        # book_..._snr-14_fileid_56.wav -> clean_fileid_56.wav
        # Keep the file-id as text so leading zeros, if any, are preserved.
        fileid_match = re.search(r"(?:^|_)fileid_(\d+)(?:_|$)", stem)
        if fileid_match is not None:
            file_id = fileid_match.group(1)
            clean_name = f"clean_fileid_{file_id}.wav"
            candidates.append(clean_root / clean_name)

            # Also support clean references stored in nested folders.
            # rglob is only used as a fallback for this one filename.
            nested_matches = sorted(clean_root.rglob(clean_name))
            candidates.extend(nested_matches)

        # Existing project convention for some DNS-style filenames.
        if "dB" in basename:
            candidates.append(clean_root / (basename.split("_")[0] + ".wav"))

        # Other common noisy_<clean-name> / noisy-<clean-name> conventions.
        for prefix in ("noisy_", "noisy-"):
            if stem.startswith(prefix):
                candidates.append(clean_root / (stem[len(prefix):] + ".wav"))

    # Preserve order but avoid reporting the same candidate repeatedly.
    seen = set()
    unique_candidates: List[Path] = []
    for p in candidates:
        key = str(p)
        if key not in seen:
            seen.add(key)
            unique_candidates.append(p)

    for p in unique_candidates:
        if p.exists():
            return p

    tried = "\n  ".join(str(p) for p in unique_candidates)
    raise FileNotFoundError(
        f"Cannot pair noisy file with clean reference:\n  noisy={noisy_path}\n"
        f"Tried:\n  {tried}\n"
        "For this DNS test set, a noisy filename containing fileid_N is expected "
        "to match clean_fileid_N.wav. Use --pairing dns (or auto)."
    )


def save_wav(root: Path, rel: Path, wav: np.ndarray, sr: int) -> None:
    path = root / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(path, np.clip(wav, -1.0, 1.0), sr)


# -----------------------------------------------------------------------------
# Metrics
# -----------------------------------------------------------------------------

def safe_pesq(clean: np.ndarray, est: np.ndarray, sr: int) -> float:
    try:
        if sr != 16000:
            clean16 = librosa.resample(clean, orig_sr=sr, target_sr=16000)
            est16 = librosa.resample(est, orig_sr=sr, target_sr=16000)
        else:
            clean16, est16 = clean, est
        return float(pesq(16000, clean16, est16, "wb"))
    except Exception:
        return float("nan")


def safe_estoi(clean: np.ndarray, est: np.ndarray, sr: int) -> float:
    try:
        return float(stoi(clean, est, sr, extended=True))
    except Exception:
        return float("nan")


def safe_energy_ratios(clean: np.ndarray, noisy: np.ndarray, est: np.ndarray) -> Tuple[float, float, float]:
    try:
        noise = noisy - clean
        vals = energy_ratios(est, clean, noise)
        return float(vals[0]), float(vals[1]), float(vals[2])
    except Exception:
        return float("nan"), float("nan"), float("nan")


def segmental_snr(clean: np.ndarray, est: np.ndarray, sr: int,
                   frame_ms: float = 20.0, hop_ratio: float = 0.5,
                   low_db: float = -10.0, high_db: float = 35.0) -> float:
    """Simple standard-style clipped segmental SNR diagnostic.

    This is included for relative diagnosis. If your paper uses another exact
    segSNR implementation, keep using that implementation for final tables.
    """
    frame = max(1, int(round(sr * frame_ms / 1000.0)))
    hop = max(1, int(round(frame * hop_ratio)))
    n = min(len(clean), len(est))
    if n < frame:
        return float("nan")
    vals = []
    eps = 1e-12
    window = np.hanning(frame).astype(np.float32)
    for s in range(0, n - frame + 1, hop):
        c = clean[s:s + frame] * window
        e = est[s:s + frame] * window
        num = float(np.sum(c * c)) + eps
        den = float(np.sum((c - e) ** 2)) + eps
        v = 10.0 * math.log10(num / den)
        vals.append(float(np.clip(v, low_db, high_db)))
    return float(np.mean(vals)) if vals else float("nan")


def compute_metrics(clean: np.ndarray, noisy: np.ndarray, est: np.ndarray, sr: int) -> Dict[str, float]:
    n = min(len(clean), len(noisy), len(est))
    clean = clean[:n]
    noisy = noisy[:n]
    est = est[:n]
    si_sdr, si_sir, si_sar = safe_energy_ratios(clean, noisy, est)
    return {
        "pesq": safe_pesq(clean, est, sr),
        "estoi": safe_estoi(clean, est, sr),
        "si_sdr": si_sdr,
        "si_sir": si_sir,
        "si_sar": si_sar,
        "seg_snr": segmental_snr(clean, est, sr),
        "wave_mse": float(np.mean((est - clean) ** 2)),
    }


# -----------------------------------------------------------------------------
# Model-domain spectral helpers
# -----------------------------------------------------------------------------

def wave_to_model_spec(model: JointSemambaSgmseModel, wav: np.ndarray, norm: float,
                       device: torch.device) -> torch.Tensor:
    """Return compressed complex model spectrum [F,T], normalized by noisy norm."""
    x = torch.from_numpy(wav).float().to(device) / max(float(norm), 1e-12)
    x = x.unsqueeze(0)  # [1,T]
    spec = model._forward_transform(model._stft(x))  # [1,F,T]
    return spec.squeeze(0)


def model_spec_to_wave(model: JointSemambaSgmseModel, spec: torch.Tensor, norm: float,
                       target_len: int) -> np.ndarray:
    wav = model.to_audio(spec, length=target_len) * float(norm)
    return wav.squeeze().detach().cpu().float().numpy()


def local_mean(x: torch.Tensor, kf: int, kt: int) -> torch.Tensor:
    """Project-matched stride-1 patch AvgPool, input [F,T], output [F,T]."""
    z = x[None, None, ...]
    z = F.avg_pool2d(
        z,
        kernel_size=(kf, kt),
        stride=1,
        padding=(kf // 2, kt // 2),
        count_include_pad=False,
    )
    return z[0, 0]


def oracle_spectra(
    X: torch.Tensor,
    Fspec: torch.Tensor,
    Bspec: torch.Tensor,
    kf: int,
    kt: int,
    eps: float,
) -> Tuple[Dict[str, torch.Tensor], Dict[str, float]]:
    """Build global, local hard, and local convex-projection oracle spectra."""
    # Pointwise complex-domain squared errors, matching the project's TF geometry.
    err_f_tf = 0.5 * torch.abs(Fspec - X).float().square()
    err_b_tf = 0.5 * torch.abs(Bspec - X).float().square()

    err_f_patch = local_mean(err_f_tf, kf, kt)
    err_b_patch = local_mean(err_b_tf, kf, kt)

    # Conservative whole-utterance oracle.
    global_f = float(err_f_tf.mean().item())
    global_b = float(err_b_tf.mean().item())
    use_front_utt = global_f < global_b
    utt = Fspec if use_front_utt else Bspec

    # Local hard oracle: choose whichever local patch error is smaller.
    front_wins = err_f_patch < err_b_patch
    hard_mask = front_wins.to(Fspec.real.dtype)
    patch = Bspec + hard_mask.to(Bspec.dtype) * (Fspec - Bspec)

    # Stronger convex-projection upper bound.
    # Locally solve min_alpha || B + alpha(F-B) - X ||^2, alpha in [0,1].
    D = Fspec - Bspec
    R = X - Bspec
    numerator_tf = torch.real(torch.conj(D) * R).float()
    denominator_tf = torch.abs(D).float().square()
    numerator = local_mean(numerator_tf, kf, kt)
    denominator = local_mean(denominator_tf, kf, kt)
    alpha = (numerator / denominator.clamp_min(eps)).clamp(0.0, 1.0)
    projection = Bspec + alpha.to(Bspec.dtype) * D

    # Diagnostics.
    eps_stat = 1e-12
    ratio = err_b_patch / err_f_patch.clamp_min(eps_stat)
    positive = ratio[front_wins]
    stats = {
        "frontend_global_better": float(use_front_utt),
        "frontend_patch_win_fraction": float(front_wins.float().mean().item()),
        "frontend_bin_win_fraction": float((err_f_tf < err_b_tf).float().mean().item()),
        "backend_to_front_patch_error_ratio_mean": float(ratio.mean().item()),
        "backend_to_front_patch_error_ratio_max": float(ratio.max().item()),
        "backend_to_front_ratio_on_front_wins_mean": float(positive.mean().item()) if positive.numel() else float("nan"),
        "projection_alpha_mean": float(alpha.mean().item()),
        "projection_alpha_active_fraction": float((alpha > 1e-4).float().mean().item()),
        "projection_alpha_front_heavy_fraction": float((alpha > 0.5).float().mean().item()),
        "frontend_spec_mse": global_f,
        "backend_spec_mse": global_b,
        "hard_oracle_spec_mse": float((0.5 * torch.abs(patch - X).float().square()).mean().item()),
        "projection_oracle_spec_mse": float((0.5 * torch.abs(projection - X).float().square()).mean().item()),
    }
    return {
        "oracle_utt": utt,
        "oracle_patch": patch,
        "oracle_projection": projection,
    }, stats


# -----------------------------------------------------------------------------
# Model loading and inference
# -----------------------------------------------------------------------------

def load_joint_model(args) -> JointSemambaSgmseModel:
    load_kwargs = dict(
        map_location="cpu",
        strict=not args.non_strict,
        data_module_cls=JointWaveDataModule,
        base_dir=".",
        batch_size=1,
        num_workers=0,
        dummy=True,
    )
    # Only explicit overrides are applied; otherwise checkpoint hparams win.
    if args.frontend is not None:
        load_kwargs["frontend"] = args.frontend
    if args.frontend_root is not None:
        load_kwargs["frontend_root"] = args.frontend_root
    if args.frontend_config is not None:
        load_kwargs["frontend_config"] = args.frontend_config
    if args.frontend_ckpt is not None:
        load_kwargs["frontend_ckpt"] = args.frontend_ckpt

    model = JointSemambaSgmseModel.load_from_checkpoint(args.ckpt, **load_kwargs)
    model.eval(no_ema=not args.use_ema).to(args.device)
    return model


def infer_frontend_backend(
    model: JointSemambaSgmseModel,
    noisy: np.ndarray,
    device: torch.device,
    args,
) -> Tuple[np.ndarray, np.ndarray]:
    y = torch.from_numpy(noisy).float().to(device)[None, None, :]  # [1,1,T]
    T = int(y.size(-1))
    norm_t = y.abs().amax(dim=-1, keepdim=True).clamp_min(1e-12)

    with torch.inference_mode():
        y_pre = model._frontend_forward(y / norm_t) * norm_t
        y_pre = match_torch_length(y_pre, T)

        train_type = model.train_type if args.train_type == "auto" else args.train_type
        if train_type == "mode2":
            y_in = y_pre.squeeze(0).squeeze(0)
            y_cond = y.squeeze(0).squeeze(0)
        elif train_type == "mode3":
            y_in = y.squeeze(0).squeeze(0)
            y_cond = y_pre.squeeze(0).squeeze(0)
        else:
            raise ValueError(f"Unsupported train_type={train_type}")

        enhanced = model.enhance(
            y_in,
            y_cond=y_cond,
            sampler_type=args.sampler_type,
            predictor="reverse_diffusion",
            corrector=args.corrector,
            N=args.N,
            snr=args.snr,
            regen_strength=args.regen_strength,
        )

    frontend = y_pre.squeeze().detach().cpu().float().numpy()
    backend = np.asarray(enhanced, dtype=np.float32).reshape(-1)
    frontend = match_np_length(frontend, T)
    backend = match_np_length(backend, T)
    # Match the existing joint_enhance_by_dir.py evaluation convention.
    frontend = np.clip(frontend, -1.0, 1.0)
    backend = np.clip(backend, -1.0, 1.0)
    return frontend, backend


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Oracle frontend/backend complementarity upper-bound analysis"
    )
    p.add_argument("--ckpt", type=str, required=True, help="No-selector joint Lightning checkpoint")
    p.add_argument("--noisy_dir", type=str, required=True)
    p.add_argument("--clean_dir", type=str, required=True)
    p.add_argument("--out_dir", type=str, required=True)
    p.add_argument("--pairing", choices=("auto", "exact", "dns"), default="auto")
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--sample_rate", type=int, default=16000)
    p.add_argument("--max_files", type=int, default=0, help="0 = all files")
    p.add_argument("--seed", type=int, default=0)

    # Inference settings.
    p.add_argument("--train_type", choices=("auto", "mode2", "mode3"), default="auto")
    p.add_argument("--sampler_type", type=str, default="pc")
    p.add_argument("--corrector", type=str, default="none")
    p.add_argument("--N", type=int, default=30)
    p.add_argument("--snr", type=float, default=0.5)
    p.add_argument("--regen_strength", type=str, default=None)
    p.add_argument("--use_ema", action="store_true")
    p.add_argument("--non_strict", action="store_true")

    # Optional frontend override; normally leave unset so checkpoint settings are used.
    p.add_argument("--frontend", choices=("semamba", "storm", "tfgridnet"), default=None)
    p.add_argument("--frontend_root", type=str, default=None)
    p.add_argument("--frontend_config", type=str, default=None)
    p.add_argument("--frontend_ckpt", type=str, default=None)

    # Oracle geometry: defaults match your current frontend-relative patch detector.
    p.add_argument("--patch_freq", type=int, default=5)
    p.add_argument("--patch_time", type=int, default=3)
    p.add_argument("--eps", type=float, default=1e-8)
    p.add_argument("--no_metrics", action="store_true", help="Only generate wavs/statistics; skip PESQ/eSTOI/etc.")
    return p


def summarize_metrics(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    metric_cols = ["pesq", "estoi", "si_sdr", "si_sir", "si_sar", "seg_snr", "wave_mse"]
    for method, g in df.groupby("method"):
        row = {"method": method, "n": len(g)}
        for m in metric_cols:
            vals = g[m].to_numpy(dtype=float)
            vals = vals[np.isfinite(vals)]
            row[m + "_mean"] = float(np.mean(vals)) if len(vals) else float("nan")
            row[m + "_std"] = float(np.std(vals)) if len(vals) else float("nan")
        rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    args = build_parser().parse_args()
    out_root = Path(args.out_dir)
    out_root.mkdir(parents=True, exist_ok=True)
    noisy_root = Path(args.noisy_dir)
    clean_root = Path(args.clean_dir)
    device = torch.device(args.device)

    set_seed(args.seed)
    print("[1/4] Loading checkpoint...")
    model = load_joint_model(args)
    print(f"  checkpoint: {args.ckpt}")
    print(f"  checkpoint train_type: {model.train_type}")
    print(f"  frontend: {getattr(model, 'frontend_name', 'unknown')}")
    print(f"  residual_refine_enable: {getattr(model, 'residual_refine_enable', False)}")
    print(f"  backend EMA: {'ON' if args.use_ema else 'OFF'}")

    wavs = list_wavs(args.noisy_dir)
    if args.max_files > 0:
        wavs = wavs[:args.max_files]
    if not wavs:
        raise RuntimeError(f"No .wav files found under {args.noisy_dir}")
    print(f"[2/4] Found {len(wavs)} test utterances")

    method_names = ["noisy", "frontend", "backend", "oracle_utt", "oracle_patch", "oracle_projection"]
    metric_rows: List[Dict[str, object]] = []
    stat_rows: List[Dict[str, object]] = []

    for idx, noisy_path in enumerate(wavs):
        rel = noisy_path.relative_to(noisy_root)
        clean_path = resolve_clean_path(noisy_path, noisy_root, clean_root, args.pairing)
        noisy = load_mono(str(noisy_path), args.sample_rate)
        clean = load_mono(str(clean_path), args.sample_rate)
        clean = match_np_length(clean, len(noisy))

        # Make backend stochasticity reproducible per utterance.
        set_seed(args.seed + idx)
        frontend, backend = infer_frontend_backend(model, noisy, device, args)

        norm = max(float(np.max(np.abs(noisy))), 1e-12)
        with torch.inference_mode():
            X = wave_to_model_spec(model, clean, norm, device)
            Fs = wave_to_model_spec(model, frontend, norm, device)
            Bs = wave_to_model_spec(model, backend, norm, device)
            oracles, stats = oracle_spectra(
                X, Fs, Bs,
                kf=args.patch_freq,
                kt=args.patch_time,
                eps=args.eps,
            )
            oracle_wavs = {
                name: np.clip(
                    model_spec_to_wave(model, spec, norm, len(noisy)), -1.0, 1.0
                )
                for name, spec in oracles.items()
            }
            # Keep the utterance-level oracle maximally conservative: when the
            # global oracle chooses a model, reuse that model's waveform directly
            # instead of adding an unnecessary STFT -> iSTFT round trip.
            oracle_wavs["oracle_utt"] = (
                frontend.copy() if stats["frontend_global_better"] > 0.5 else backend.copy()
            )

        waves = {
            "noisy": np.clip(noisy, -1.0, 1.0),
            "frontend": frontend,
            "backend": backend,
            **oracle_wavs,
        }

        # Save all outputs so your existing DNSMOS/WSS/CSIG/CBAK/COVL evaluator can be reused unchanged.
        for method, wav in waves.items():
            save_wav(out_root / method, rel, wav, args.sample_rate)

        stat_row: Dict[str, object] = {"filename": str(rel), **stats}
        # Direct upper-bound gains in the actual evaluation wave domain.
        stat_row["frontend_wave_mse"] = float(np.mean((frontend - clean) ** 2))
        stat_row["backend_wave_mse"] = float(np.mean((backend - clean) ** 2))
        for name, wav in oracle_wavs.items():
            stat_row[name + "_wave_mse"] = float(np.mean((wav - clean) ** 2))
        stat_rows.append(stat_row)

        if not args.no_metrics:
            for method, wav in waves.items():
                metrics = compute_metrics(clean, noisy, wav, args.sample_rate)
                metric_rows.append({"filename": str(rel), "method": method, **metrics})

        print(
            f"[{idx + 1:4d}/{len(wavs)}] {rel} | "
            f"front-win patch={100.0 * stats['frontend_patch_win_fraction']:.2f}% | "
            f"ratio_max={stats['backend_to_front_patch_error_ratio_max']:.2f} | "
            f"alpha_mean={stats['projection_alpha_mean']:.3f}"
        )

    print("[3/4] Writing diagnostics...")
    stat_df = pd.DataFrame(stat_rows)
    stat_df.to_csv(out_root / "oracle_stats.csv", index=False)

    if not args.no_metrics:
        metric_df = pd.DataFrame(metric_rows)
        metric_df.to_csv(out_root / "per_file_metrics.csv", index=False)
        summary = summarize_metrics(metric_df)
        summary.to_csv(out_root / "summary.csv", index=False)
        print("\n=== Metric summary ===")
        with pd.option_context("display.max_columns", None, "display.width", 240):
            print(summary.to_string(index=False))

        # A compact go/no-go table against the backend baseline.
        piv = metric_df.pivot(index="filename", columns="method")
        delta_rows = []
        for method in ("frontend", "oracle_utt", "oracle_patch", "oracle_projection"):
            row = {"method_vs_backend": method}
            for metric in ("pesq", "estoi", "si_sdr", "seg_snr"):
                try:
                    d = (piv[metric][method] - piv[metric]["backend"]).to_numpy(float)
                    d = d[np.isfinite(d)]
                    row["delta_" + metric + "_mean"] = float(d.mean()) if len(d) else float("nan")
                    row["delta_" + metric + "_std"] = float(d.std()) if len(d) else float("nan")
                except Exception:
                    row["delta_" + metric + "_mean"] = float("nan")
                    row["delta_" + metric + "_std"] = float("nan")
            delta_rows.append(row)
        delta_df = pd.DataFrame(delta_rows)
        delta_df.to_csv(out_root / "oracle_gain_vs_backend.csv", index=False)
        print("\n=== Oracle gain vs backend ===")
        print(delta_df.to_string(index=False))

    # Dataset-level selector diagnostics.
    stats_summary = {
        "n_files": int(len(stat_df)),
        "frontend_global_better_fraction": float(stat_df["frontend_global_better"].mean()),
        "frontend_patch_win_fraction_mean": float(stat_df["frontend_patch_win_fraction"].mean()),
        "frontend_bin_win_fraction_mean": float(stat_df["frontend_bin_win_fraction"].mean()),
        "backend_to_front_patch_error_ratio_mean": float(stat_df["backend_to_front_patch_error_ratio_mean"].mean()),
        "backend_to_front_patch_error_ratio_max_over_dataset": float(stat_df["backend_to_front_patch_error_ratio_max"].max()),
        "projection_alpha_mean": float(stat_df["projection_alpha_mean"].mean()),
        "projection_alpha_active_fraction_mean": float(stat_df["projection_alpha_active_fraction"].mean()),
        "projection_alpha_front_heavy_fraction_mean": float(stat_df["projection_alpha_front_heavy_fraction"].mean()),
    }
    with open(out_root / "oracle_stats_summary.json", "w", encoding="utf-8") as f:
        json.dump(stats_summary, f, indent=2, ensure_ascii=False)

    run_info = f"""Oracle Complementarity Upper-Bound Experiment
================================================
checkpoint: {args.ckpt}
noisy_dir: {args.noisy_dir}
clean_dir: {args.clean_dir}
N: {args.N}
corrector: {args.corrector}
snr: {args.snr}
use_ema: {args.use_ema}
patch: {args.patch_freq} x {args.patch_time}
seed: {args.seed}

IMPORTANT
---------
This experiment deliberately uses clean speech to construct oracle decisions.
It measures complementarity / headroom only. Oracle outputs are NOT deployable.

Interpretation
--------------
- oracle_utt: conservative whole-utterance complementarity upper bound.
- oracle_patch: local hard frontend-vs-backend selection upper bound.
- oracle_projection: stronger local convex interpolation upper bound.
- If oracle_patch/projection clearly improve final metrics over backend, the
  frontend/backend cooperation idea has real headroom and the missing piece is
  an inference-time reliability estimator/controller.
- If even the oracle gains are tiny, further frontend-guidance losses are
  unlikely to provide a large global gain on this test distribution.

The generated wav directories can be passed directly to your existing external
DNSMOS / WSS / CSIG / CBAK / COVL evaluator, which is not included in this zip.
"""
    (out_root / "RUN_INFO.txt").write_text(run_info, encoding="utf-8")

    print("\n[4/4] Done.")
    print(f"Outputs: {out_root}")
    print("Key files:")
    print(f"  {out_root / 'oracle_stats.csv'}")
    if not args.no_metrics:
        print(f"  {out_root / 'summary.csv'}")
        print(f"  {out_root / 'oracle_gain_vs_backend.csv'}")
    print(f"  {out_root / 'oracle_stats_summary.json'}")


if __name__ == "__main__":
    main()
