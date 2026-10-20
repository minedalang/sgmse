#!/usr/bin/env python3
"""Evaluation-only oracle diagnostic for Future-RRSD harm calibration."""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from functools import partial
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _read_wav(path: Path):
    x, sr = sf.read(str(path), dtype="float32", always_2d=False)
    x = np.asarray(x)
    if x.ndim > 1:
        x = x.mean(axis=-1)
    return torch.from_numpy(x), int(sr)


def _clean_for_file(noisy_path: Path, clean_dir: Path) -> Path:
    m = re.search(r"fileid_(\d+)", noisy_path.name)
    if m is None:
        raise FileNotFoundError(f"cannot parse fileid from {noisy_path.name}")
    return clean_dir / f"clean_fileid_{m.group(1)}.wav"


def _match_length(x: torch.Tensor, n: int) -> torch.Tensor:
    if x.numel() == n:
        return x
    if x.numel() > n:
        start = (x.numel() - n) // 2
        return x[start:start + n]
    return F.pad(x, (0, n - x.numel()))


def _load_model(args):
    from sgmse.joint_data_module import JointWaveDataModule
    from sgmse.joint_model import JointSemambaSgmseModel

    model = JointSemambaSgmseModel.load_from_checkpoint(
        args.ckpt,
        map_location="cpu",
        strict=True,
        data_module_cls=JointWaveDataModule,
        base_dir=".", batch_size=1, num_workers=0, dummy=True,
        weights_only=False,
    )
    model.eval(no_ema=not args.use_ema).to(args.device)
    return model


def _prepare_inputs(model, noisy: torch.Tensor, device: str):
    y = noisy.to(device).float()[None, None, :]
    norm = y.abs().amax(dim=-1, keepdim=True).clamp_min(1e-12)
    y_pre = model._frontend_forward(y / norm) * norm
    if y_pre.shape[-1] >= y.shape[-1]:
        y_pre = y_pre[..., :y.shape[-1]]
    else:
        y_pre = F.pad(y_pre, (0, y.shape[-1] - y_pre.shape[-1]))
    if model.train_type == "mode2":
        return y_pre.squeeze(0), y.squeeze(0)
    return y.squeeze(0), y_pre.squeeze(0)


def _make_specs(model, y_main, y_cond, device):
    from sgmse.util.other import pad_spec

    ref = y_cond if model.train_type == "mode2" else y_main
    norm = ref.abs().max().clamp_min(1e-12)
    y_main = y_main.to(device) / norm
    y_cond = y_cond.to(device) / norm
    Y = pad_spec(torch.unsqueeze(model._forward_transform(model._stft(y_main)), 0))
    C = pad_spec(torch.unsqueeze(model._forward_transform(model._stft(y_cond)), 0))
    return Y, C, norm


def _wavlm_features(wavlm, wav: torch.Tensor):
    wav = wav.float()
    wav = wav - wav.mean(dim=-1, keepdim=True)
    wav = wav / wav.abs().amax(dim=-1, keepdim=True).clamp_min(1e-6)
    return wavlm(input_values=wav, output_hidden_states=True, return_dict=True).hidden_states[3]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--noisy_dir", required=True)
    p.add_argument("--clean_dir", required=True)
    p.add_argument("--ckpt", default="joint_train_logs/sb61i0gj_RRSD/epoch=118-step=80000.ckpt")
    p.add_argument("--out_dir", default="experiment_logs/rrsd_harm_oracle_smoke")
    p.add_argument("--max_files", type=int, default=0)
    p.add_argument("--harm_threshold", type=float, default=0.0)
    p.add_argument("--harm_scale", type=float, default=0.0)
    p.add_argument("--N", type=int, default=30)
    p.add_argument("--snr", type=float, default=0.5)
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--use_ema", action="store_true")
    p.add_argument("--wavlm_model", default="microsoft/wavlm-base-plus")
    args = p.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    model = _load_model(args)
    from transformers import WavLMModel
    wavlm = WavLMModel.from_pretrained(args.wavlm_model, local_files_only=True).to(args.device).eval()
    for q in wavlm.parameters():
        q.requires_grad_(False)

    files = sorted(Path(args.noisy_dir).glob("*.wav"))
    if args.max_files > 0:
        files = files[:args.max_files]
    rows = []
    for index, noisy_path in enumerate(files):
        clean_path = _clean_for_file(noisy_path, Path(args.clean_dir))
        if not clean_path.exists():
            raise FileNotFoundError(clean_path)
        noisy, sr = _read_wav(noisy_path)
        clean, csr = _read_wav(clean_path)
        if sr != 16000 or csr != 16000:
            raise ValueError(f"oracle diagnostic expects 16 kHz: {sr}, {csr}")
        L = min(noisy.numel(), clean.numel())
        noisy, clean = noisy[:L], clean[:L]
        y_main, y_cond = _prepare_inputs(model, noisy, args.device)
        Y, C, norm = _make_specs(model, y_main, y_cond, args.device)
        score_fn = partial(model, y_cond=C)
        Y_pre, _ = model._main_and_cond_to_pre_noisy(Y, C)
        x0_front = Y_pre.detach()
        base = model.build_rrsd_score_calibrator(Y, C, scale=1.0, t_max=float(model.sde.T))
        if base is None:
            raise RuntimeError("checkpoint has no RRSD router")

        target_wav = _match_length(clean, model._spec_to_wave_batch(Y_pre).shape[-1])[None].to(args.device)
        target_feat = _wavlm_features(wavlm, target_wav).detach()
        state = {"harm_steps": 0, "steps": 0, "mse_base": [], "mse_route": []}

        def calibrator(x_t, y_unused, t, score_base, **kwargs):
            routed = base(x_t, y_unused, t, score_base, **kwargs)
            trace = getattr(base, "last_trace", None)
            if trace is None:
                return routed
            xb, xf, alpha = trace["x0_base"], trace["x0_front"], trace["alpha_tile"]
            alpha_map = F.interpolate(alpha.float(), size=xb.shape[-2:], mode="bilinear", align_corners=False)
            dtype = xb.real.dtype if torch.is_complex(xb) else xb.dtype
            xr = xb + alpha_map.to(dtype=dtype) * (xf - xb)
            wb, wr = model._spec_to_wave_batch(xb), model._spec_to_wave_batch(xr)
            mb = (_wavlm_features(wavlm, wb) - target_feat).square().mean().item()
            mr = (_wavlm_features(wavlm, wr) - target_feat).square().mean().item()
            state["steps"] += 1
            state["mse_base"].append(mb)
            state["mse_route"].append(mr)
            if mr > mb + args.harm_threshold:
                state["harm_steps"] += 1
                routed = score_base + float(args.harm_scale) * (routed - score_base)
            calibrator.last_trace = trace
            return routed

        h = hashlib.blake2b(noisy_path.name.encode(), digest_size=4).digest()
        torch.manual_seed((args.seed + int.from_bytes(h, "little")) % (2**32))
        sampler = model.get_pc_sampler(
            "reverse_diffusion", "none", Y, score_fn=score_fn, N=args.N,
            snr=args.snr, score_calibrator=calibrator,
        )
        sample, _ = sampler()
        enhanced = model.to_audio(sample.squeeze(), length=L).detach().cpu().numpy() * float(norm)
        sf.write(str(out / noisy_path.name), np.clip(enhanced, -1.0, 1.0), 16000)
        rows.append({"file": noisy_path.name, "clean": str(clean_path), **state})
        print(f"[{index + 1}/{len(files)}] {noisy_path.name} harm={state['harm_steps']}/{state['steps']}", flush=True)

    (out / "metrics.json").write_text(json.dumps(rows, indent=2) + "\n")


if __name__ == "__main__":
    main()
