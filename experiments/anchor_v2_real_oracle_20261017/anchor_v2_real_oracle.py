#!/usr/bin/env python3
"""Clean-reference diagnostic for the Anchor V2 residual subspace.

This file is deliberately independent from the deployment sampler.  It injects
a temporary controller whose coefficients are solved from the clean reference
at each predictor step.  The clean reference is therefore diagnostic-only and
is never used by the normal ``enhance`` path.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path
from typing import Dict

import numpy as np
import soundfile as sf
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sgmse.anchor_v2 import solve_two_direction_tile_oracle
from sgmse.util.other import pad_spec


def load_wav(path: str, device: str):
    wav, sr = sf.read(path, dtype="float32", always_2d=False)
    wav = np.asarray(wav)
    if wav.ndim > 1:
        wav = wav.mean(axis=-1)
    if sr != 16000:
        raise ValueError(f"expected 16 kHz, got {sr}: {path}")
    return torch.from_numpy(wav).to(device=device), sr


def match_1d(x: torch.Tensor, n: int) -> torch.Tensor:
    if x.numel() == n:
        return x
    if x.numel() > n:
        start = (x.numel() - n) // 2
        return x[start:start + n]
    return F.pad(x, (0, n - x.numel()))


def match_spec(x: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Center crop/pad a [B,C,F,T] complex spectrum to target shape."""
    out = x
    for dim in (-2, -1):
        n = target.shape[dim]
        cur = out.shape[dim]
        if cur > n:
            start = (cur - n) // 2
            sl = [slice(None)] * out.ndim
            sl[dim] = slice(start, start + n)
            out = out[tuple(sl)]
        elif cur < n:
            before = (n - cur) // 2
            after = n - cur - before
            pads = [0, 0, 0, 0]
            if dim == -1:
                pads = [before, after, 0, 0]
            else:
                pads = [0, 0, before, after]
            out = F.pad(out, pads)
    return out


class CleanOracleController(nn.Module):
    """Per-step bounded two-direction LS oracle, for diagnosis only."""

    def __init__(self, clean_spec, num_bands=32, num_time_blocks=16,
                 a_max=0.25, b_max=0.15, time_power=1.0, state=None):
        super().__init__()
        self.register_buffer("clean_spec", clean_spec.detach())
        self.num_bands = int(num_bands)
        self.num_time_blocks = int(num_time_blocks)
        self.a_max = float(a_max)
        self.b_max = float(b_max)
        self.time_power = float(time_power)
        self.state = state if state is not None else {}

    def forward(self, *, x_t, x0_base, y_pre, y_noisy, t, t_max=1.0,
                detach_inputs=True, a_max_override=None, b_max_override=None,
                scale=None, **kwargs):
        if t.dim() == 0:
            t = torch.full((x_t.shape[0],), float(t.item()), device=x_t.device)
        else:
            t = t.to(device=x_t.device, dtype=torch.float32).flatten()
        clean = self.clean_spec.to(device=x0_base.device, dtype=x0_base.dtype)
        clean = match_spec(clean, x0_base)
        residual = clean - x0_base
        a, b, stats = solve_two_direction_tile_oracle(
            residual, y_pre - x0_base, y_noisy - y_pre,
            num_bands=self.num_bands, num_time_blocks=self.num_time_blocks,
            eps=1e-6,
        )
        t_norm = (t / max(float(t_max), 1e-8)).clamp(0.0, 1.0)
        env = (4.0 * t_norm.pow(self.time_power) * (1.0 - t_norm.pow(self.time_power))).clamp(0.0, 1.0)
        scale_eff = 1.0 if scale is None else max(float(scale), 0.0)
        ac = (self.a_max if a_max_override is None else float(a_max_override)) * scale_eff
        bc = (self.b_max if b_max_override is None else float(b_max_override)) * scale_eff
        a = a.clamp(-ac, ac) * env.view(-1, 1, 1, 1)
        b = b.clamp(-bc, bc) * env.view(-1, 1, 1, 1)
        size = (x0_base.shape[-2], x0_base.shape[-1])
        amap = F.interpolate(a, size=size, mode="bilinear", align_corners=False)
        bmap = F.interpolate(b, size=size, mode="bilinear", align_corners=False)
        delta = amap.to(dtype=x0_base.real.dtype if torch.is_complex(x0_base) else x0_base.dtype) * (y_pre - x0_base)
        delta = delta + bmap.to(dtype=x0_base.real.dtype if torch.is_complex(x0_base) else x0_base.dtype) * (y_noisy - y_pre)
        self.state.setdefault("a_abs", []).append(float(a.detach().abs().mean().cpu()))
        self.state.setdefault("b_abs", []).append(float(b.detach().abs().mean().cpu()))
        self.state.setdefault("conditioning", []).append(float(stats["conditioning"].detach().mean().cpu()))
        return {
            "x0_calib": x0_base + delta,
            "delta_x0": delta,
            "delta_x0_pre": amap * (y_pre - x0_base),
            "delta_x0_noisy": bmap * (y_noisy - y_pre),
            "a_map": amap, "b_map": bmap, "a_tile": a, "b_tile": b,
            "E_t": env.view(-1, 1, 1, 1),
        }


def prepare_specs(model, noisy, clean, device):
    y = noisy[None, None, :].to(device)
    c = clean[None, None, :].to(device)
    norm = y.abs().amax().clamp_min(1e-12)
    y = y / norm
    y_pre = model._frontend_forward(y)  # [1,1,T]
    if y_pre.shape[-1] > y.shape[-1]:
        y_pre = y_pre[..., :y.shape[-1]]
    elif y_pre.shape[-1] < y.shape[-1]:
        y_pre = F.pad(y_pre, (0, y.shape[-1] - y_pre.shape[-1]))
    if model.train_type == "mode2":
        main, cond = y_pre[:, 0], y[:, 0]
    else:
        main, cond = y[:, 0], y_pre[:, 0]
    Y = pad_spec(torch.unsqueeze(model._forward_transform(model._stft(main)), 0))
    C = pad_spec(torch.unsqueeze(model._forward_transform(model._stft(cond)), 0))
    clean_n = c / norm
    X = pad_spec(torch.unsqueeze(model._forward_transform(model._stft(clean_n[:, 0])), 0))
    X = match_spec(X, Y)
    return Y, C, X, norm, int(noisy.numel())


def sample_one(model, Y, C, X, norm, length, arm, args, state):
    score_fn = lambda x, y, t: model(x, y, t, y_cond=C)
    calibrator = None
    if arm == "router":
        calibrator = model.build_rrsd_score_calibrator(Y, C, scale=1.0)
    elif arm == "tcut005":
        calibrator = model.build_rrsd_score_calibrator(Y, C, scale=1.0, t_cut=0.05, t_cut_scale=0.0)
    elif arm == "anchor_oracle":
        controller = CleanOracleController(X, args.num_bands, args.num_time_blocks,
                                            args.a_max, args.b_max, args.time_power, state)
        model.anchor_v2_controller = controller
        calibrator = model.build_anchor_v2_score_calibrator(Y, C, scale=1.0)
    torch.manual_seed(args.seed)
    sampler = model.get_pc_sampler("reverse_diffusion", "none", Y, score_fn=score_fn,
                                    N=args.N, snr=args.snr, score_calibrator=calibrator)
    sample, nfe = sampler()
    wav = model.to_audio(sample.squeeze(), length=length).detach().cpu() * float(norm.cpu())
    return wav.numpy(), int(nfe)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default="joint_train_logs/sb61i0gj_RRSD/epoch=118-step=80000.ckpt")
    p.add_argument("--manifest", default="experiments/rrsd_cv_repeat32/manifest.csv")
    p.add_argument("--out_dir", default="experiments/anchor_v2_real_oracle_20261017")
    p.add_argument("--wav_dir", default="/dev/shm/sgmse_anchor_v2_oracle")
    p.add_argument("--max_files", type=int, default=1)
    p.add_argument("--N", type=int, default=30)
    p.add_argument("--snr", type=float, default=0.5)
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--device", default="cuda")
    p.add_argument("--use_ema", action="store_true")
    p.add_argument("--num_bands", type=int, default=32)
    p.add_argument("--num_time_blocks", type=int, default=16)
    p.add_argument("--a_max", type=float, default=0.25)
    p.add_argument("--b_max", type=float, default=0.15)
    p.add_argument("--time_power", type=float, default=1.0)
    args = p.parse_args()
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    wav_dir = Path(args.wav_dir); wav_dir.mkdir(parents=True, exist_ok=True)
    from sgmse.joint_data_module import JointWaveDataModule
    from sgmse.joint_model import JointSemambaSgmseModel
    model = JointSemambaSgmseModel.load_from_checkpoint(
        args.ckpt, map_location="cpu", strict=True, data_module_cls=JointWaveDataModule,
        base_dir=".", batch_size=1, num_workers=0, dummy=True, weights_only=False)
    model.eval(no_ema=not args.use_ema).to(args.device)
    with open(args.manifest, newline="") as f:
        rows = list(csv.DictReader(f))[:args.max_files]
    result_rows = []
    for row in rows:
        noisy, nsr = load_wav(row["noisy"], args.device)
        clean, csr = load_wav(row["clean"], args.device)
        length = min(noisy.numel(), clean.numel())
        noisy, clean = noisy[:length], clean[:length]
        Y, C, X, norm, length = prepare_specs(model, noisy, clean, args.device)
        h = hashlib.blake2b(row["filename"].encode(), digest_size=4).digest()
        file_seed = (int(args.seed) + int.from_bytes(h, "little")) % (2**32)
        for arm in ("router", "tcut005", "anchor_oracle"):
            state: Dict = {}
            args.seed = file_seed
            with torch.inference_mode():
                wav, nfe = sample_one(model, Y, C, X, norm, length, arm, args, state)
            path = wav_dir / f"{Path(row['filename']).stem}_{arm}.wav"
            sf.write(path, np.clip(wav, -1.0, 1.0), 16000)
            result_rows.append({"file": row["filename"], "arm": arm, "nfe": nfe,
                                "wav": str(path), "a_abs_mean": np.mean(state.get("a_abs", [0.0])),
                                "b_abs_mean": np.mean(state.get("b_abs", [0.0])),
                                "conditioning_mean": np.mean(state.get("conditioning", [0.0]))})
            print(json.dumps(result_rows[-1]), flush=True)
    (out / "smoke_results.json").write_text(json.dumps(result_rows, indent=2) + "\n")
    print(f"Wrote {out / 'smoke_results.json'}; temporary wavs are under {wav_dir}")


if __name__ == "__main__":
    main()
