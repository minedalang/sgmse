#!/usr/bin/env python3
"""
End-to-end diagnostic inference for the exported joint StoRM + SGMSE ONNX pair.

This script intentionally uses ONNX Runtime only for the two neural networks,
while reproducing the reference SGMSE audio preprocessing and OUVE-PC sampler
in Python/NumPy:

  noisy wav
    -> normalize(noisy peak)
    -> torch.stft(center=True, periodic Hann)
    -> exponent complex transform
    -> storm_frontend.onnx
    -> pre-enhanced spectrum
    -> OUVE prior centered on pre-enhanced
    -> PC sampling (ALD corrector + reverse-diffusion predictor)
    -> inverse transform / torch.istft
    -> de-normalize

Important for this exporter:
  * sgmse_score.onnx already returns the FINAL score, including network_scaling=1/t.
    Do NOT divide its output by t again.
  * train_type=mode2 means OUVE's mean/observation is the pre-enhanced spectrum.
    The original noisy spectrum is still passed to the score ONNX as a condition.
  * PyTorch complex randn has real/imag component variance 1/2.  The RI noise here
    therefore uses N(0, 1/sqrt(2)) independently in both channels.

For a fixed-frame ONNX (normally T=256), the script tests one chunk whose waveform
length is (T-1)*hop_length samples.  Longer files are center-cropped by default;
use --start-seconds to select another chunk.  Shorter files are symmetrically
zero-padded.  This makes the test deterministic and close to the training framing.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np


DEFAULT_CFG: Dict[str, Any] = {
    "sample_rate": 16000,
    "n_fft": 510,
    "hop_length": 128,
    "window": "hann",
    "center": True,
    "transform_type": "exponent",
    "spec_factor": 0.15,
    "spec_abs_exponent": 0.5,
    "normalization": "noisy",
    "theta": 1.5,
    "sigma_min": 0.05,
    "sigma_max": 0.5,
    "N": 30,
    "t_eps": 0.03,
    "sampler_type": "pc",
    "train_type": "mode2",
}


@dataclass
class StepStats:
    step: int
    t: float
    h: float
    score_rms: float
    score_max: float
    x_rms: float
    x_max: float


def _require_deps():
    try:
        import onnxruntime as ort  # noqa: F401
    except ImportError as exc:
        raise RuntimeError("Missing dependency: pip install onnxruntime") from exc
    try:
        import torch  # noqa: F401
    except ImportError as exc:
        raise RuntimeError("Missing dependency: pip install torch") from exc
    try:
        import soundfile as sf  # noqa: F401
    except ImportError as exc:
        raise RuntimeError("Missing dependency: pip install soundfile") from exc
    try:
        import scipy.signal  # noqa: F401
    except ImportError as exc:
        raise RuntimeError("Missing dependency: pip install scipy") from exc


def _load_manifest(onnx_dir: Path, manifest_path: Path | None) -> Tuple[Dict[str, Any], Dict[str, Any] | None]:
    cfg = dict(DEFAULT_CFG)
    manifest: Dict[str, Any] | None = None
    p = manifest_path if manifest_path is not None else onnx_dir / "model_manifest.json"
    if p.is_file():
        manifest = json.loads(p.read_text(encoding="utf-8"))
        audio = manifest.get("audio", {})
        ouve = manifest.get("ouve", {})
        cfg.update({
            "sample_rate": int(audio.get("sample_rate", cfg["sample_rate"])),
            "n_fft": int(audio.get("n_fft", cfg["n_fft"])),
            "hop_length": int(audio.get("hop_length", cfg["hop_length"])),
            "window": str(audio.get("window", cfg["window"])),
            "center": bool(audio.get("center", cfg["center"])),
            "transform_type": str(audio.get("transform_type", cfg["transform_type"])),
            "spec_factor": float(audio.get("spec_factor", cfg["spec_factor"])),
            "spec_abs_exponent": float(audio.get("spec_abs_exponent", cfg["spec_abs_exponent"])),
            "normalization": str(audio.get("normalization", cfg["normalization"])),
            "theta": float(ouve.get("theta", cfg["theta"])),
            "sigma_min": float(ouve.get("sigma_min", cfg["sigma_min"])),
            "sigma_max": float(ouve.get("sigma_max", cfg["sigma_max"])),
            "N": int(ouve.get("N", cfg["N"])),
            "t_eps": float(ouve.get("t_eps", cfg["t_eps"])),
            "sampler_type": str(ouve.get("sampler_type", cfg["sampler_type"])),
            "train_type": str(ouve.get("train_type", cfg["train_type"])),
        })
        print(f"[manifest] {p}")
        print(f"[manifest] exported weights = {manifest.get('weights', 'unknown')}")
    else:
        print(f"[manifest] not found: {p}; using checkpoint-known defaults", file=sys.stderr)
    return cfg, manifest


def _make_session(path: Path, provider: str):
    import onnxruntime as ort

    if not path.is_file():
        raise FileNotFoundError(path)
    available = ort.get_available_providers()
    if provider == "cuda":
        if "CUDAExecutionProvider" not in available:
            raise RuntimeError(f"CUDAExecutionProvider unavailable; available={available}")
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    else:
        providers = ["CPUExecutionProvider"]
    opts = ort.SessionOptions()
    opts.log_severity_level = 3
    return ort.InferenceSession(str(path), sess_options=opts, providers=providers)


def _static_dim(v: Any, label: str) -> int:
    if isinstance(v, (int, np.integer)) and int(v) > 0:
        return int(v)
    raise RuntimeError(f"{label} must be a fixed positive dimension in this diagnostic exporter, got {v!r}")


def _inspect_onnx_shapes(front_sess, score_sess) -> Tuple[int, int]:
    fi = front_sess.get_inputs()[0]
    shape = fi.shape
    if len(shape) != 4:
        raise RuntimeError(f"frontend input must be rank-4 [B,2,F,T], got {shape}")
    if shape[1] != 2:
        raise RuntimeError(f"frontend channel dimension must be 2 (real/imag), got {shape}")
    f_bins = _static_dim(shape[2], "frequency bins")
    frames = _static_dim(shape[3], "time frames")

    expected_score_inputs = {x.name for x in score_sess.get_inputs()}
    required = {"x_t_ri", "preenhanced_spec_ri", "noisy_spec_ri", "t"}
    if not required.issubset(expected_score_inputs):
        raise RuntimeError(f"score ONNX inputs={sorted(expected_score_inputs)}, expected at least {sorted(required)}")
    print(f"[onnx] frontend input: {fi.name} {shape}")
    print(f"[onnx] score inputs: {[f'{x.name}:{x.shape}' for x in score_sess.get_inputs()]}")
    print(f"[onnx] fixed spectrogram size: F={f_bins}, T={frames}")
    return f_bins, frames


def _read_audio(path: Path, target_sr: int) -> Tuple[np.ndarray, int]:
    import soundfile as sf
    from scipy.signal import resample_poly

    wav, sr = sf.read(str(path), dtype="float32", always_2d=False)
    wav = np.asarray(wav, dtype=np.float32)
    if wav.ndim == 2:
        wav = wav.mean(axis=1, dtype=np.float32)
    if wav.ndim != 1 or wav.size == 0:
        raise RuntimeError(f"invalid audio shape: {wav.shape}")
    if sr != target_sr:
        g = math.gcd(int(sr), int(target_sr))
        wav = resample_poly(wav, target_sr // g, sr // g).astype(np.float32, copy=False)
        print(f"[audio] resampled {sr} -> {target_sr} Hz")
        sr = target_sr
    if not np.all(np.isfinite(wav)):
        raise RuntimeError("input audio contains NaN/Inf")
    return wav, sr


def _select_fixed_chunk(wav: np.ndarray, target_len: int, sr: int, start_seconds: float | None):
    original_len = len(wav)
    if original_len >= target_len:
        if start_seconds is None:
            start = (original_len - target_len) // 2
        else:
            start = int(round(start_seconds * sr))
            start = max(0, min(start, original_len - target_len))
        chunk = wav[start:start + target_len].copy()
        crop = (0, target_len)
        desc = f"crop input [{start}:{start + target_len}] ({start/sr:.3f}s .. {(start+target_len)/sr:.3f}s)"
        return chunk, crop, desc

    pad = target_len - original_len
    left = pad // 2
    right = pad - left
    chunk = np.pad(wav, (left, right), mode="constant").astype(np.float32, copy=False)
    crop = (left, left + original_len)
    desc = f"symmetric zero pad left={left}, right={right}"
    return chunk, crop, desc


def _window(n_fft: int, kind: str):
    import torch

    if kind == "hann":
        return torch.hann_window(n_fft, periodic=True, dtype=torch.float32)
    if kind == "sqrthann":
        return torch.sqrt(torch.hann_window(n_fft, periodic=True, dtype=torch.float32))
    raise ValueError(f"unsupported window: {kind}")


def _spec_forward(spec, transform_type: str, factor: float, exponent: float):
    import torch

    if transform_type == "exponent":
        if exponent != 1.0:
            spec = spec.abs().pow(exponent) * torch.exp(1j * torch.angle(spec))
        return spec * factor
    if transform_type == "log":
        return torch.log1p(spec.abs()) * torch.exp(1j * torch.angle(spec)) * factor
    if transform_type == "none":
        return spec
    raise ValueError(f"unsupported transform_type={transform_type!r}")


def _spec_backward(spec, transform_type: str, factor: float, exponent: float):
    import torch

    if transform_type == "exponent":
        spec = spec / factor
        if exponent != 1.0:
            spec = spec.abs().pow(1.0 / exponent) * torch.exp(1j * torch.angle(spec))
        return spec
    if transform_type == "log":
        spec = spec / factor
        return torch.expm1(spec.abs()) * torch.exp(1j * torch.angle(spec))
    if transform_type == "none":
        return spec
    raise ValueError(f"unsupported transform_type={transform_type!r}")


def _wav_to_ri(wav_norm: np.ndarray, cfg: Dict[str, Any]) -> np.ndarray:
    import torch

    x = torch.from_numpy(wav_norm).float().unsqueeze(0)  # [1,L]
    win = _window(cfg["n_fft"], cfg["window"])
    spec = torch.stft(
        x,
        n_fft=cfg["n_fft"],
        hop_length=cfg["hop_length"],
        window=win,
        center=cfg["center"],
        return_complex=True,
    )  # [1,F,T] (mono channel)
    spec = _spec_forward(spec, cfg["transform_type"], cfg["spec_factor"], cfg["spec_abs_exponent"])
    spec = spec.unsqueeze(0)  # [B=1,C=1,F,T]
    ri = torch.cat([spec.real, spec.imag], dim=1).contiguous().numpy().astype(np.float32)
    return ri


def _ri_to_wav(ri: np.ndarray, length: int, cfg: Dict[str, Any]) -> np.ndarray:
    import torch

    r = torch.from_numpy(np.asarray(ri[:, 0:1], dtype=np.float32))
    i = torch.from_numpy(np.asarray(ri[:, 1:2], dtype=np.float32))
    spec = torch.complex(r, i).squeeze(0).squeeze(0)  # [F,T]
    spec = _spec_backward(spec, cfg["transform_type"], cfg["spec_factor"], cfg["spec_abs_exponent"])
    win = _window(cfg["n_fft"], cfg["window"])
    wav = torch.istft(
        spec,
        n_fft=cfg["n_fft"],
        hop_length=cfg["hop_length"],
        window=win,
        center=cfg["center"],
        length=length,
    )
    return wav.detach().cpu().numpy().astype(np.float32, copy=False)


def _rms(x: np.ndarray) -> float:
    x64 = np.asarray(x, dtype=np.float64)
    return float(np.sqrt(np.mean(x64 * x64)))


def _describe(label: str, x: np.ndarray) -> None:
    x = np.asarray(x)
    print(
        f"[{label}] shape={tuple(x.shape)} rms={_rms(x):.6g} "
        f"mean={float(np.mean(x)):.6g} min={float(np.min(x)):.6g} max={float(np.max(x)):.6g}"
    )


def _ouve_std(t: float | np.ndarray, theta: float, sigma_min: float, sigma_max: float) -> np.ndarray:
    t = np.asarray(t, dtype=np.float64)
    logsig = math.log(sigma_max / sigma_min)
    inside = (
        sigma_min**2
        * np.exp(-2.0 * theta * t)
        * (np.exp(2.0 * (theta + logsig) * t) - 1.0)
        * logsig
        / (theta + logsig)
    )
    return np.sqrt(np.maximum(inside, 0.0))


def _ouve_diffusion(t: float, sigma_min: float, sigma_max: float) -> float:
    logsig = math.log(sigma_max / sigma_min)
    sigma = sigma_min * (sigma_max / sigma_min) ** float(t)
    return float(sigma * math.sqrt(2.0 * logsig))


def _complex_randn_ri(rng: np.random.Generator, shape: Tuple[int, ...]) -> np.ndarray:
    # Match torch.randn_like(complex64): real and imag each have variance 1/2.
    return (rng.standard_normal(shape, dtype=np.float32) / np.float32(math.sqrt(2.0))).astype(np.float32)


def _score(score_sess, x: np.ndarray, pre: np.ndarray, noisy: np.ndarray, t: float) -> np.ndarray:
    # IMPORTANT: exported ONNX has already applied network_scaling=1/t.
    out = score_sess.run(
        None,
        {
            "x_t_ri": np.asarray(x, dtype=np.float32),
            "preenhanced_spec_ri": np.asarray(pre, dtype=np.float32),
            "noisy_spec_ri": np.asarray(noisy, dtype=np.float32),
            "t": np.asarray([t], dtype=np.float32),
        },
    )[0]
    out = np.asarray(out, dtype=np.float32)
    if not np.all(np.isfinite(out)):
        raise FloatingPointError(f"score ONNX produced NaN/Inf at t={t}")
    return out


def _pc_sample(
    score_sess,
    pre: np.ndarray,
    noisy: np.ndarray,
    cfg: Dict[str, Any],
    N: int,
    eps: float,
    snr: float,
    corrector_steps: int,
    seed: int,
    log_every: int,
) -> Tuple[np.ndarray, List[StepStats]]:
    if N <= 0:
        raise ValueError("N must be positive")
    if not (0.0 < eps <= 1.0):
        raise ValueError("eps must be in (0,1]")
    if corrector_steps < 0:
        raise ValueError("corrector_steps must be >= 0")

    theta = float(cfg["theta"])
    sigma_min = float(cfg["sigma_min"])
    sigma_max = float(cfg["sigma_max"])
    rng = np.random.default_rng(seed)

    # OUVESDE.prior_sampling: x_T = y + std(T) * complex N(0,1), with y=pre for mode2.
    std_T = float(_ouve_std(1.0, theta, sigma_min, sigma_max))
    x = pre + np.float32(std_T) * _complex_randn_ri(rng, tuple(pre.shape))
    x = np.asarray(x, dtype=np.float32)
    print(f"[sampler] OUVE prior centered on PRE-ENHANCED, std(T=1)={std_T:.8g}")
    _describe("sampler/x_T", x)

    timesteps = np.linspace(1.0, eps, N, dtype=np.float64)
    stats: List[StepStats] = []
    x_mean = x

    for idx, t64 in enumerate(timesteps):
        t = float(t64)
        h = float(timesteps[idx] - timesteps[idx + 1]) if idx != N - 1 else float(timesteps[-1])

        # ALD corrector, matching SGMSE's AnnealedLangevinDynamics.
        std_t = float(_ouve_std(t, theta, sigma_min, sigma_max))
        step_size = 2.0 * (snr * std_t) ** 2
        for _ in range(corrector_steps):
            grad = _score(score_sess, x, pre, noisy, t)
            noise = _complex_randn_ri(rng, tuple(x.shape))
            x_mean = x + np.float32(step_size) * grad
            x = x_mean + np.float32(math.sqrt(2.0 * step_size)) * noise

        # ReverseDiffusionPredictor:
        # forward discrete: f = theta*(y-x)*h, G = g(t)*sqrt(h)
        # reverse discrete: rev_f = f - G^2*score
        # predictor mean: x_mean = x - rev_f
        score = _score(score_sess, x, pre, noisy, t)
        g = _ouve_diffusion(t, sigma_min, sigma_max)
        G = g * math.sqrt(h)
        f = np.float32(theta * h) * (pre - x)
        rev_f = f - np.float32(G * G) * score
        x_mean = x - rev_f
        x = x_mean + np.float32(G) * _complex_randn_ri(rng, tuple(x.shape))

        st = StepStats(
            step=idx + 1,
            t=t,
            h=h,
            score_rms=_rms(score),
            score_max=float(np.max(np.abs(score))),
            x_rms=_rms(x),
            x_max=float(np.max(np.abs(x))),
        )
        stats.append(st)

        if idx == 0 or idx == N - 1 or ((idx + 1) % max(1, log_every) == 0):
            print(
                f"[step {idx+1:02d}/{N}] t={t:.6f} h={h:.6f} "
                f"score_rms={st.score_rms:.5g} score_max={st.score_max:.5g} "
                f"x_rms={st.x_rms:.5g} x_max={st.x_max:.5g}"
            )
        if not np.all(np.isfinite(x)):
            raise FloatingPointError(f"sampler state became NaN/Inf at step {idx+1}, t={t}")
        if st.x_max > 1e4:
            print(f"[WARNING] sampler magnitude is exploding at step {idx+1}: max={st.x_max:.3g}", file=sys.stderr)

    # Reference PC sampler uses final predictor mean when denoise=True.
    return np.asarray(x_mean, dtype=np.float32), stats


def _write_wav(path: Path, wav: np.ndarray, sr: int) -> None:
    import soundfile as sf

    path.parent.mkdir(parents=True, exist_ok=True)
    wav = np.asarray(wav, dtype=np.float32)
    if not np.all(np.isfinite(wav)):
        raise FloatingPointError(f"refusing to write non-finite audio: {path}")
    # WAV FLOAT avoids hiding explosions via implicit PCM clipping during diagnosis.
    sf.write(str(path), wav, sr, subtype="FLOAT")


def main() -> None:
    parser = argparse.ArgumentParser(description="Diagnose joint StoRM+SGMSE ONNX with reference OUVE-PC inference")
    parser.add_argument("--onnx-dir", type=Path, required=True, help="Directory containing storm_frontend.onnx and sgmse_score.onnx")
    parser.add_argument("--input", type=Path, required=True, help="Input noisy WAV")
    parser.add_argument("--output", type=Path, required=True, help="Final enhanced WAV")
    parser.add_argument("--manifest", type=Path, default=None, help="Optional model_manifest.json; defaults to <onnx-dir>/model_manifest.json")
    parser.add_argument("--provider", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--steps", type=int, default=None, help="Override PC sampling steps N (default from manifest, normally 30)")
    parser.add_argument("--corrector-steps", type=int, default=1, help="ALD corrector iterations per time step")
    parser.add_argument("--snr", type=float, default=0.5, help="ALD SNR (reference enhance default: 0.5)")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--start-seconds", type=float, default=None, help="For long WAVs, start time of the tested fixed-size chunk; default=center crop")
    parser.add_argument("--debug-dir", type=Path, default=None, help="Directory for frontend-only WAV and JSON stats")
    parser.add_argument("--log-every", type=int, default=3)
    args = parser.parse_args()

    _require_deps()
    cfg, manifest = _load_manifest(args.onnx_dir, args.manifest)

    if cfg["normalization"] != "noisy":
        raise RuntimeError(f"this diagnostic expects normalize='noisy', manifest says {cfg['normalization']!r}")
    if cfg["train_type"] != "mode2":
        raise RuntimeError(f"this diagnostic is intentionally for train_type='mode2', got {cfg['train_type']!r}")
    if cfg["sampler_type"] != "pc":
        print(f"[WARNING] manifest sampler_type={cfg['sampler_type']!r}, but this script runs PC for diagnosis", file=sys.stderr)
    if manifest is not None and manifest.get("weights") == "ema":
        print("[WARNING] manifest says this ONNX was exported with EMA. For your known-good test, re-export with --no-ema.", file=sys.stderr)

    front_path = args.onnx_dir / "storm_frontend.onnx"
    score_path = args.onnx_dir / "sgmse_score.onnx"
    front_sess = _make_session(front_path, args.provider)
    score_sess = _make_session(score_path, args.provider)
    f_bins, frames = _inspect_onnx_shapes(front_sess, score_sess)

    expected_f = cfg["n_fft"] // 2 + 1
    if f_bins != expected_f:
        raise RuntimeError(f"ONNX F={f_bins}, but n_fft={cfg['n_fft']} implies F={expected_f}")

    target_len = (frames - 1) * cfg["hop_length"]  # exact for center=True
    wav, sr = _read_audio(args.input, cfg["sample_rate"])
    chunk, crop, desc = _select_fixed_chunk(wav, target_len, sr, args.start_seconds)
    print(f"[audio] input samples={len(wav)}, test samples={target_len} ({target_len/sr:.3f}s): {desc}")

    norm_factor = float(np.max(np.abs(chunk)))
    if not np.isfinite(norm_factor) or norm_factor <= 1e-12:
        raise RuntimeError(f"input chunk has invalid/near-zero peak: {norm_factor}")
    wav_norm = chunk / np.float32(norm_factor)
    print(f"[audio] noisy peak normalization factor={norm_factor:.8g}")

    noisy_ri = _wav_to_ri(wav_norm, cfg)
    if noisy_ri.shape != (1, 2, f_bins, frames):
        raise RuntimeError(f"preprocessed spectrogram shape={noisy_ri.shape}, ONNX expects (1,2,{f_bins},{frames})")
    _describe("noisy_spec_ri", noisy_ri)

    print("[frontend] running storm_frontend.onnx")
    pre_ri = np.asarray(front_sess.run(None, {front_sess.get_inputs()[0].name: noisy_ri})[0], dtype=np.float32)
    if pre_ri.shape != noisy_ri.shape:
        raise RuntimeError(f"frontend output shape {pre_ri.shape} != input shape {noisy_ri.shape}")
    if not np.all(np.isfinite(pre_ri)):
        raise FloatingPointError("frontend ONNX produced NaN/Inf")
    _describe("preenhanced_spec_ri", pre_ri)

    debug_dir = args.debug_dir if args.debug_dir is not None else args.output.parent / (args.output.stem + "_debug")
    debug_dir.mkdir(parents=True, exist_ok=True)

    frontend_wav_full = _ri_to_wav(pre_ri, target_len, cfg) * np.float32(norm_factor)
    c0, c1 = crop
    frontend_wav = frontend_wav_full[c0:c1]
    frontend_path = debug_dir / "frontend_only.wav"
    _write_wav(frontend_path, frontend_wav, sr)
    print(f"[frontend] wrote {frontend_path}")
    _describe("frontend_audio", frontend_wav)

    # Probe score at several t values before the stochastic sampler.
    probe_x = pre_ri.copy()
    probe_stats: Dict[str, Dict[str, float]] = {}
    for t in (1.0, 0.5, 0.1, float(cfg["t_eps"])):
        s = _score(score_sess, probe_x, pre_ri, noisy_ri, t)
        probe_stats[f"{t:.6g}"] = {"rms": _rms(s), "max_abs": float(np.max(np.abs(s)))}
        print(f"[score probe] t={t:.6g}: rms={_rms(s):.6g}, max_abs={float(np.max(np.abs(s))):.6g}")

    N = int(args.steps if args.steps is not None else cfg["N"])
    print(
        f"[sampler] PC reverse_diffusion + ALD: N={N}, eps={cfg['t_eps']}, "
        f"snr={args.snr}, corrector_steps={args.corrector_steps}, seed={args.seed}"
    )
    sample_ri, step_stats = _pc_sample(
        score_sess=score_sess,
        pre=pre_ri,
        noisy=noisy_ri,
        cfg=cfg,
        N=N,
        eps=float(cfg["t_eps"]),
        snr=float(args.snr),
        corrector_steps=int(args.corrector_steps),
        seed=int(args.seed),
        log_every=int(args.log_every),
    )
    _describe("final_spec_ri", sample_ri)

    out_full = _ri_to_wav(sample_ri, target_len, cfg) * np.float32(norm_factor)
    out = out_full[c0:c1]
    _write_wav(args.output, out, sr)
    _describe("final_audio", out)
    print(f"[done] wrote final: {args.output}")

    report = {
        "onnx_dir": str(args.onnx_dir),
        "input": str(args.input),
        "output": str(args.output),
        "manifest_weights": None if manifest is None else manifest.get("weights"),
        "config": cfg,
        "onnx_shape": {"freq_bins": f_bins, "frames": frames},
        "test_chunk_samples": target_len,
        "normalization_factor": norm_factor,
        "score_probe": probe_stats,
        "sampler": {
            "steps": N,
            "eps": float(cfg["t_eps"]),
            "snr": float(args.snr),
            "corrector_steps": int(args.corrector_steps),
            "seed": int(args.seed),
            "nfe": N * (int(args.corrector_steps) + 1),
        },
        "step_stats": [asdict(x) for x in step_stats],
        "frontend_audio": str(frontend_path),
    }
    report_path = debug_dir / "diagnostic.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[done] diagnostic: {report_path}")
    print("\nInterpretation:")
    print("  1) frontend_only.wav already bad -> frontend export/preprocessing/weights problem.")
    print("  2) frontend_only.wav good, final bad -> score ONNX or OUVE sampling/full-ScoreModel logic problem.")
    print("  3) both good here, Android bad -> Android preprocessing/sampler/RI-layout implementation problem.")
    print("  4) This script intentionally does NOT invent custom score_reweight/denoiser-to-score logic that is not inside the exported ONNX.")


if __name__ == "__main__":
    main()
