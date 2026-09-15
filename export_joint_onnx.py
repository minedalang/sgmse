#!/usr/bin/env python3
"""Export the supplied joint StoRM + SGMSE checkpoint to Android-friendly ONNX.

The checkpoint used by this project contains two neural networks:
  1. StoRM frontend: transformed noisy spectrum -> transformed pre-enhanced spectrum
  2. NCSN++ score network: (x_t, pre-enhanced, noisy, t) -> OUVE score

PyTorch complex tensors are deliberately not exposed to ONNX. Every complex
spectrum is represented as float32 [B, 2, F, T], channel 0 = real and channel
1 = imaginary.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn


def _load_checkpoint(path: Path) -> Mapping[str, object]:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:  # torch < 2.0
        return torch.load(path, map_location="cpu")


def _require_equal(hp: Mapping[str, object], expected: Mapping[str, object]) -> None:
    errors = []
    for key, want in expected.items():
        got = hp.get(key)
        if got != want:
            errors.append(f"{key}: checkpoint={got!r}, exporter requires={want!r}")
    if errors:
        raise RuntimeError(
            "This real-only exporter does not match the checkpoint architecture:\n  - "
            + "\n  - ".join(errors)
        )


def _split_prefixed_state(
    state: Mapping[str, torch.Tensor], prefix: str
) -> Dict[str, torch.Tensor]:
    out = {k[len(prefix) :]: v for k, v in state.items() if k.startswith(prefix)}
    if not out:
        raise RuntimeError(f"No state_dict entries found with prefix {prefix!r}")
    return out


def _load_strict(module: nn.Module, state: Mapping[str, torch.Tensor], label: str) -> None:
    result = module.load_state_dict(state, strict=True)
    if result.missing_keys or result.unexpected_keys:
        raise RuntimeError(
            f"{label} state mismatch: missing={result.missing_keys}, "
            f"unexpected={result.unexpected_keys}"
        )


def _tracked_parameters(
    backend: nn.Module, frontend: nn.Module, frontend_trainable: bool
) -> List[Tuple[str, nn.Parameter]]:
    params = [(f"dnn.{n}", p) for n, p in backend.named_parameters() if p.requires_grad]
    if frontend_trainable:
        params.extend(
            (f"frontend.{n}", p)
            for n, p in frontend.named_parameters()
            if p.requires_grad
        )
    return params


def _apply_ema(
    checkpoint: Mapping[str, object],
    backend: nn.Module,
    frontend: nn.Module,
    frontend_trainable: bool,
) -> List[str]:
    ema = checkpoint.get("ema")
    if not isinstance(ema, Mapping):
        raise RuntimeError("--use-ema was requested, but checkpoint['ema'] is absent")
    shadow = ema.get("shadow_params")
    if not isinstance(shadow, (list, tuple)):
        raise RuntimeError("checkpoint['ema']['shadow_params'] is absent or invalid")

    tracked = _tracked_parameters(backend, frontend, frontend_trainable)
    if len(tracked) != len(shadow):
        raise RuntimeError(
            f"EMA tensor count mismatch: architecture has {len(tracked)} tracked parameters, "
            f"checkpoint has {len(shadow)} shadow tensors"
        )

    names: List[str] = []
    with torch.no_grad():
        for (name, param), src in zip(tracked, shadow):
            if not isinstance(src, torch.Tensor):
                raise RuntimeError(f"EMA entry for {name} is not a Tensor")
            if tuple(param.shape) != tuple(src.shape):
                raise RuntimeError(
                    f"EMA shape mismatch at {name}: model={tuple(param.shape)}, "
                    f"checkpoint={tuple(src.shape)}"
                )
            param.copy_(src.to(device=param.device, dtype=param.dtype))
            names.append(name)
    return names


class StormRealWrapper(nn.Module):
    """Real-channel version of the StoRM NCSN++ frontend backbone."""

    def __init__(self, backbone: nn.Module, time_cond: float):
        super().__init__()
        self.backbone = backbone
        self.time_cond = float(time_cond)

    def forward(self, noisy_spec_ri: torch.Tensor) -> torch.Tensor:
        # noisy_spec_ri: [B,2,F,T], already STFT-transformed and padded.
        net = self.backbone
        modules = net.all_modules
        m_idx = 0
        x = noisy_spec_ri
        time_cond = torch.ones_like(x[:, 0, 0, 0]) * self.time_cond

        if net.embedding_type == "fourier":
            used_sigmas = time_cond
            temb = modules[m_idx](torch.log(used_sigmas))
            m_idx += 1
        elif net.embedding_type == "positional":
            from sgmse.backbones.ncsnpp_utils import layers

            used_sigmas = net.sigmas[time_cond.long()]
            temb = layers.get_timestep_embedding(time_cond, net.nf)
        else:
            raise RuntimeError(f"Unsupported frontend embedding_type={net.embedding_type!r}")

        if net.conditional:
            temb = modules[m_idx](temb)
            m_idx += 1
            temb = modules[m_idx](net.act(temb))
            m_idx += 1
        else:
            temb = None

        if not net.centered:
            x = 2.0 * x - 1.0

        input_pyramid = x if net.progressive_input != "none" else None
        hs = [modules[m_idx](x)]
        m_idx += 1

        for i_level in range(net.num_resolutions):
            for _ in range(net.num_res_blocks):
                h = modules[m_idx](hs[-1], temb)
                m_idx += 1
                if h.shape[-2] in net.attn_resolutions:
                    h = modules[m_idx](h)
                    m_idx += 1
                hs.append(h)

            if i_level != net.num_resolutions - 1:
                if net.resblock_type == "ddpm":
                    h = modules[m_idx](hs[-1])
                else:
                    h = modules[m_idx](hs[-1], temb)
                m_idx += 1

                if net.progressive_input == "input_skip":
                    input_pyramid = net.pyramid_downsample(input_pyramid)
                    h = modules[m_idx](input_pyramid, h)
                    m_idx += 1
                elif net.progressive_input == "residual":
                    input_pyramid = modules[m_idx](input_pyramid)
                    m_idx += 1
                    if net.skip_rescale:
                        input_pyramid = (input_pyramid + h) / math.sqrt(2.0)
                    else:
                        input_pyramid = input_pyramid + h
                    h = input_pyramid
                hs.append(h)

        h = hs[-1]
        h = modules[m_idx](h, temb)
        m_idx += 1
        h = modules[m_idx](h)
        m_idx += 1
        h = modules[m_idx](h, temb)
        m_idx += 1

        pyramid = None
        for i_level in reversed(range(net.num_resolutions)):
            for _ in range(net.num_res_blocks + 1):
                h = modules[m_idx](torch.cat([h, hs.pop()], dim=1), temb)
                m_idx += 1

            if h.shape[-2] in net.attn_resolutions:
                h = modules[m_idx](h)
                m_idx += 1

            if net.progressive != "none":
                if i_level == net.num_resolutions - 1:
                    pyramid = net.act(modules[m_idx](h))
                    m_idx += 1
                    pyramid = modules[m_idx](pyramid)
                    m_idx += 1
                elif net.progressive == "output_skip":
                    pyramid = net.pyramid_upsample(pyramid)
                    pyramid_h = net.act(modules[m_idx](h))
                    m_idx += 1
                    pyramid_h = modules[m_idx](pyramid_h)
                    m_idx += 1
                    pyramid = pyramid + pyramid_h
                elif net.progressive == "residual":
                    pyramid = modules[m_idx](pyramid)
                    m_idx += 1
                    if net.skip_rescale:
                        pyramid = (pyramid + h) / math.sqrt(2.0)
                    else:
                        pyramid = pyramid + h
                    h = pyramid
                else:
                    raise RuntimeError(f"Unsupported progressive={net.progressive!r}")

            if i_level != 0:
                h = modules[m_idx](h) if net.resblock_type == "ddpm" else modules[m_idx](h, temb)
                m_idx += 1

        if hs:
            raise RuntimeError("Frontend skip stack was not fully consumed")
        if net.progressive == "output_skip":
            h = pyramid
        else:
            h = net.act(modules[m_idx](h))
            m_idx += 1
            h = modules[m_idx](h)
            m_idx += 1

        if m_idx != len(modules):
            raise RuntimeError(f"Frontend module cursor mismatch: {m_idx} != {len(modules)}")
        if net.scale_by_sigma:
            scale = used_sigmas.reshape((x.shape[0], 1, 1, 1))
            h = h / scale

        # Original code converts these two channels to complex. Keep them real.
        return net.output_layer(h)


class ScoreRealWrapper(nn.Module):
    """Real-channel NCSN++ score network for this checkpoint's exact mode."""

    def __init__(self, dnn: nn.Module):
        super().__init__()
        self.dnn = dnn
        unsupported = {
            "cross_attn": bool(dnn.cross_attn),
            "film_modulate": bool(dnn.film_modulate),
            "use_tfgrid_selector": bool(dnn.use_tfgrid_selector),
            "use_residual_regen_selector": bool(dnn.use_residual_regen_selector),
            "use_noise_prior_guidance_selector": bool(dnn.use_noise_prior_guidance_selector),
        }
        enabled = [k for k, v in unsupported.items() if v]
        if enabled:
            raise RuntimeError("This checkpoint enables unsupported branches: " + ", ".join(enabled))

    def forward(
        self,
        x_t_ri: torch.Tensor,
        preenhanced_spec_ri: torch.Tensor,
        noisy_spec_ri: torch.Tensor,
        t: torch.Tensor,
    ) -> torch.Tensor:
        net = self.dnn
        modules = net.all_modules
        m_idx = 0

        # Original complex order for mode2_include_noisy_main:
        # x.real,x.imag,noisy.real,noisy.imag,pre.real,pre.imag
        x = torch.cat([x_t_ri, noisy_spec_ri, preenhanced_spec_ri], dim=1)
        x = 2.0 * x - 1.0

        if net.embedding_type == "fourier":
            temb = modules[m_idx](torch.log(t))
            m_idx += 1
        elif net.embedding_type == "positional":
            from sgmse.backbones.ncsnpp_utils import layers

            temb = layers.get_timestep_embedding(t, net.nf)
        else:
            raise RuntimeError(f"Unsupported backend embedding_type={net.embedding_type!r}")
        temb = modules[m_idx](temb)
        m_idx += 1
        temb = modules[m_idx](net.act(temb))
        m_idx += 1

        input_pyramid = x if net.progressive_input != "none" else None
        hs = [modules[m_idx](x)]
        m_idx += 1

        for i_level in range(net.num_resolutions):
            for _ in range(net.num_res_blocks):
                h = modules[m_idx](hs[-1], temb)
                m_idx += 1
                if h.shape[-2] in net.attn_resolutions:
                    h = modules[m_idx](h)
                    m_idx += 1
                hs.append(h)

            if i_level != net.num_resolutions - 1:
                if net.resblock_type == "ddpm":
                    h = modules[m_idx](hs[-1])
                else:
                    h = modules[m_idx](hs[-1], temb)
                m_idx += 1

                if net.progressive_input == "input_skip":
                    input_pyramid = net.pyramid_downsample(input_pyramid)
                    h = modules[m_idx](input_pyramid, h)
                    m_idx += 1
                elif net.progressive_input == "residual":
                    input_pyramid = modules[m_idx](input_pyramid)
                    m_idx += 1
                    if net.skip_rescale:
                        input_pyramid = (input_pyramid + h) / math.sqrt(2.0)
                    else:
                        input_pyramid = input_pyramid + h
                    h = input_pyramid
                hs.append(h)

        h = hs[-1]
        h = modules[m_idx](h, temb)
        m_idx += 1
        h = modules[m_idx](h)
        m_idx += 1
        h = modules[m_idx](h, temb)
        m_idx += 1

        pyramid = None
        for i_level in reversed(range(net.num_resolutions)):
            for _ in range(net.num_res_blocks + 1):
                h = modules[m_idx](torch.cat([h, hs.pop()], dim=1), temb)
                m_idx += 1

            if h.shape[-2] in net.attn_resolutions:
                h = modules[m_idx](h)
                m_idx += 1

            if net.progressive != "none":
                if i_level == net.num_resolutions - 1:
                    pyramid = net.act(modules[m_idx](h))
                    m_idx += 1
                    pyramid = modules[m_idx](pyramid)
                    m_idx += 1
                elif net.progressive == "output_skip":
                    pyramid = net.pyramid_upsample(pyramid)
                    pyramid_h = net.act(modules[m_idx](h))
                    m_idx += 1
                    pyramid_h = modules[m_idx](pyramid_h)
                    m_idx += 1
                    pyramid = pyramid + pyramid_h
                elif net.progressive == "residual":
                    pyramid = modules[m_idx](pyramid)
                    m_idx += 1
                    if net.skip_rescale:
                        pyramid = (pyramid + h) / math.sqrt(2.0)
                    else:
                        pyramid = pyramid + h
                    h = pyramid
                else:
                    raise RuntimeError(f"Unsupported progressive={net.progressive!r}")

            if i_level != 0:
                h = modules[m_idx](h) if net.resblock_type == "ddpm" else modules[m_idx](h, temb)
                m_idx += 1

        if hs:
            raise RuntimeError("Backend skip stack was not fully consumed")
        if net.progressive == "output_skip":
            h = pyramid
        else:
            h = net.act(modules[m_idx](h))
            m_idx += 1
            h = modules[m_idx](h)
            m_idx += 1
        if m_idx != len(modules):
            raise RuntimeError(f"Backend module cursor mismatch: {m_idx} != {len(modules)}")

        f_ri = net.output_layer(h)
        # Checkpoint: score_matching, c_skip=0, c_out=1, network_scaling=1/t.
        return f_ri / t[:, None, None, None]


def _complex_from_ri(x: torch.Tensor) -> torch.Tensor:
    return torch.complex(x[:, 0:1], x[:, 1:2])


def _ri_from_complex(x: torch.Tensor) -> torch.Tensor:
    return torch.cat([x.real, x.imag], dim=1)


def _max_error(a: torch.Tensor, b: torch.Tensor) -> Tuple[float, float]:
    diff = (a.float() - b.float()).abs()
    abs_err = float(diff.max().item())
    rel_err = float((diff / b.float().abs().clamp_min(1e-5)).max().item())
    return abs_err, rel_err


def _pytorch_equivalence_check(
    frontend: nn.Module,
    backend: nn.Module,
    frontend_wrapper: StormRealWrapper,
    backend_wrapper: ScoreRealWrapper,
    batch: int,
    freq_bins: int,
    frames: int,
) -> Dict[str, float]:
    torch.manual_seed(1234)
    noisy = torch.randn(batch, 2, freq_bins, frames) * 0.1
    pre = torch.randn_like(noisy) * 0.1
    xt = torch.randn_like(noisy) * 0.1
    t = torch.linspace(0.15, 0.85, batch)

    with torch.inference_mode():
        front_ref_c = frontend.backbone(
            _complex_from_ri(noisy),
            torch.full((batch,), float(frontend.storm_time_cond)),
        )
        front_ref = _ri_from_complex(front_ref_c)
        front_got = frontend_wrapper(noisy)

        backend_ref_c = backend(
            _complex_from_ri(xt),
            _complex_from_ri(pre),
            t,
            y_cond=_complex_from_ri(noisy),
            y_extra=_complex_from_ri(noisy),
        )
        backend_ref = _ri_from_complex(backend_ref_c) / t[:, None, None, None]
        backend_got = backend_wrapper(xt, pre, noisy, t)

    f_abs, f_rel = _max_error(front_got, front_ref)
    b_abs, b_rel = _max_error(backend_got, backend_ref)
    if f_abs > 2e-4 or b_abs > 2e-4:
        raise RuntimeError(
            "Real wrapper does not match original PyTorch complex forward: "
            f"frontend abs={f_abs:.3e}, backend abs={b_abs:.3e}"
        )
    return {
        "frontend_max_abs": f_abs,
        "frontend_max_rel": f_rel,
        "backend_max_abs": b_abs,
        "backend_max_rel": b_rel,
    }


def _export_onnx(
    module: nn.Module,
    args: Tuple[torch.Tensor, ...],
    path: Path,
    input_names: Sequence[str],
    output_names: Sequence[str],
    opset: int,
    dynamic_batch: bool,
) -> None:
    dynamic_axes = None
    if dynamic_batch:
        dynamic_axes = {name: {0: "batch"} for name in [*input_names, *output_names]}
    torch.onnx.export(
        module,
        args,
        str(path),
        export_params=True,
        opset_version=opset,
        do_constant_folding=True,
        input_names=list(input_names),
        output_names=list(output_names),
        dynamic_axes=dynamic_axes,
    )


def _onnx_check(paths: Iterable[Path]) -> None:
    try:
        import onnx
    except ImportError as exc:
        raise RuntimeError("Install 'onnx' to export and validate the models") from exc
    for path in paths:
        onnx.checker.check_model(onnx.load(str(path)))


def _ort_check(
    frontend_path: Path,
    backend_path: Path,
    front: StormRealWrapper,
    back: ScoreRealWrapper,
    batch: int,
    freq_bins: int,
    frames: int,
) -> Dict[str, float]:
    try:
        import onnxruntime as ort
    except ImportError as exc:
        raise RuntimeError("Install 'onnxruntime' or pass --skip-ort-check") from exc

    rng = np.random.default_rng(5678)
    noisy = (rng.standard_normal((batch, 2, freq_bins, frames)) * 0.1).astype(np.float32)
    pre = (rng.standard_normal(noisy.shape) * 0.1).astype(np.float32)
    xt = (rng.standard_normal(noisy.shape) * 0.1).astype(np.float32)
    t = np.linspace(0.15, 0.85, batch, dtype=np.float32)

    front_sess = ort.InferenceSession(str(frontend_path), providers=["CPUExecutionProvider"])
    back_sess = ort.InferenceSession(str(backend_path), providers=["CPUExecutionProvider"])
    front_ort = front_sess.run(None, {"noisy_spec_ri": noisy})[0]
    back_ort = back_sess.run(
        None,
        {
            "x_t_ri": xt,
            "preenhanced_spec_ri": pre,
            "noisy_spec_ri": noisy,
            "t": t,
        },
    )[0]
    with torch.inference_mode():
        front_pt = front(torch.from_numpy(noisy)).numpy()
        back_pt = back(
            torch.from_numpy(xt),
            torch.from_numpy(pre),
            torch.from_numpy(noisy),
            torch.from_numpy(t),
        ).numpy()
    f_abs = float(np.max(np.abs(front_ort - front_pt)))
    b_abs = float(np.max(np.abs(back_ort - back_pt)))
    if f_abs > 2e-3 or b_abs > 2e-3:
        raise RuntimeError(
            f"ONNX Runtime mismatch: frontend max abs={f_abs:.3e}, "
            f"backend max abs={b_abs:.3e}"
        )
    return {"frontend_ort_max_abs": f_abs, "backend_ort_max_abs": b_abs}


def _build_modules(project_root: Path, checkpoint: Mapping[str, object]):
    sys.path.insert(0, str(project_root.resolve()))
    from sgmse.backbones import BackboneRegistry
    from sgmse.frontends.storm import StormFrontend

    hp = checkpoint.get("hyper_parameters")
    state = checkpoint.get("state_dict")
    if not isinstance(hp, Mapping) or not isinstance(state, Mapping):
        raise RuntimeError("Expected a Lightning checkpoint with hyper_parameters and state_dict")

    _require_equal(
        hp,
        {
            "backbone": "ncsnpp_v2",
            "sde": "ouve",
            "train_type": "mode2",
            "loss_type": "score_matching",
            "loss_weighting": "sigma^2",
            "network_scaling": "1/t",
            "c_in": "1",
            "c_out": "1",
            "c_skip": "0",
            "mode2_include_noisy_main": True,
            "frontend": "storm",
            "cross_attn": False,
            "film_modulate": False,
            "selector_type": "none",
            "residual_refine_enable": False,
        },
    )

    dnn_cls = BackboneRegistry.get_by_name("ncsnpp_v2")
    dnn_kwargs = dict(hp)
    dnn_kwargs["mode2_include_noisy_main"] = True
    backend = dnn_cls(**dnn_kwargs)
    frontend = StormFrontend(**dict(hp))

    _load_strict(backend, _split_prefixed_state(state, "dnn."), "backend")
    _load_strict(frontend, _split_prefixed_state(state, "frontend."), "frontend")
    return hp, frontend, backend


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export this project's joint StoRM + SGMSE checkpoint to ONNX"
    )
    parser.add_argument("--ckpt", type=Path, required=True, help="Large Lightning .ckpt on your machine")
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path(__file__).resolve().parent.parent,
        help="Directory containing the sgmse/ package",
    )
    parser.add_argument("--out-dir", type=Path, default=Path("onnx_out"))
    parser.add_argument("--frames", type=int, default=256, help="Fixed padded STFT frame count; multiple of 64")
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--opset", type=int, default=18)
    parser.add_argument("--dynamic-batch", action="store_true")
    parser.add_argument("--no-ema", action="store_true", help="Export raw state_dict instead of checkpoint EMA")
    parser.add_argument("--skip-pytorch-check", action="store_true")
    parser.add_argument("--skip-ort-check", action="store_true")
    args = parser.parse_args()

    if not args.ckpt.is_file():
        raise FileNotFoundError(args.ckpt)
    if not (args.project_root / "sgmse").is_dir():
        raise FileNotFoundError(f"sgmse package not found under {args.project_root}")
    if args.frames <= 0 or args.frames % 64 != 0:
        raise ValueError("--frames must be a positive multiple of 64")
    if args.batch <= 0:
        raise ValueError("--batch must be positive")

    print(f"[1/6] Loading checkpoint: {args.ckpt}", flush=True)
    checkpoint = _load_checkpoint(args.ckpt)
    print("[2/6] Rebuilding frontend and backend from checkpoint hyperparameters", flush=True)
    hp, frontend, backend = _build_modules(args.project_root, checkpoint)

    ema_names: List[str] = []
    if not args.no_ema:
        ema_names = _apply_ema(
            checkpoint,
            backend,
            frontend,
            bool(hp.get("frontend_trainable", False)),
        )
        print(f"      Applied EMA to {len(ema_names)} parameter tensors", flush=True)
    else:
        print("      Using raw state_dict weights (--no-ema)", flush=True)

    frontend.eval()
    backend.eval()
    front_wrapper = StormRealWrapper(frontend.backbone, float(hp.get("storm_time_cond", 1.0))).eval()
    back_wrapper = ScoreRealWrapper(backend).eval()

    freq_bins = int(hp["n_fft"]) // 2 + 1
    check_stats: Dict[str, float] = {}
    if not args.skip_pytorch_check:
        print("[3/6] Checking real wrappers against original complex PyTorch", flush=True)
        check_stats.update(
            _pytorch_equivalence_check(
                frontend,
                backend,
                front_wrapper,
                back_wrapper,
                args.batch,
                freq_bins,
                args.frames,
            )
        )
    else:
        print("[3/6] PyTorch equivalence check skipped", flush=True)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    frontend_path = args.out_dir / "storm_frontend.onnx"
    backend_path = args.out_dir / "sgmse_score.onnx"
    dummy = torch.zeros(args.batch, 2, freq_bins, args.frames, dtype=torch.float32)
    dummy_t = torch.full((args.batch,), 0.5, dtype=torch.float32)

    print(f"[4/6] Exporting {frontend_path}", flush=True)
    _export_onnx(
        front_wrapper,
        (dummy,),
        frontend_path,
        ["noisy_spec_ri"],
        ["preenhanced_spec_ri"],
        args.opset,
        args.dynamic_batch,
    )
    print(f"      Exporting {backend_path}", flush=True)
    _export_onnx(
        back_wrapper,
        (dummy, dummy, dummy, dummy_t),
        backend_path,
        ["x_t_ri", "preenhanced_spec_ri", "noisy_spec_ri", "t"],
        ["score_ri"],
        args.opset,
        args.dynamic_batch,
    )

    print("[5/6] Running ONNX structural checker", flush=True)
    _onnx_check([frontend_path, backend_path])
    if not args.skip_ort_check:
        print("      Comparing ONNX Runtime against PyTorch", flush=True)
        check_stats.update(
            _ort_check(
                frontend_path,
                backend_path,
                front_wrapper,
                back_wrapper,
                args.batch,
                freq_bins,
                args.frames,
            )
        )
    else:
        print("      ONNX Runtime numerical check skipped", flush=True)

    manifest = {
        "checkpoint": str(args.ckpt),
        "weights": "raw" if args.no_ema else "ema",
        "opset": args.opset,
        "models": {
            "storm_frontend.onnx": {
                "inputs": {"noisy_spec_ri": ["B", 2, freq_bins, args.frames]},
                "outputs": {"preenhanced_spec_ri": ["B", 2, freq_bins, args.frames]},
            },
            "sgmse_score.onnx": {
                "inputs": {
                    "x_t_ri": ["B", 2, freq_bins, args.frames],
                    "preenhanced_spec_ri": ["B", 2, freq_bins, args.frames],
                    "noisy_spec_ri": ["B", 2, freq_bins, args.frames],
                    "t": ["B"],
                },
                "outputs": {"score_ri": ["B", 2, freq_bins, args.frames]},
            },
        },
        "complex_layout": "channel 0 = real, channel 1 = imaginary",
        "audio": {
            "sample_rate": int(hp["sr"]),
            "n_fft": int(hp["n_fft"]),
            "hop_length": int(hp["hop_length"]),
            "window": str(hp["window"]),
            "center": True,
            "transform_type": str(hp["transform_type"]),
            "spec_factor": float(hp["spec_factor"]),
            "spec_abs_exponent": float(hp["spec_abs_exponent"]),
            "normalization": str(hp["normalize"]),
            "pad_frames_to_multiple": 64,
        },
        "ouve": {
            "theta": float(hp["theta"]),
            "sigma_min": float(hp["sigma_min"]),
            "sigma_max": float(hp["sigma_max"]),
            "N": int(hp["N"]),
            "t_eps": float(hp["t_eps"]),
            "sampler_type": str(hp["sampler_type"]),
            "train_type": str(hp["train_type"]),
        },
        "checks": check_stats,
        "ema_parameter_count": len(ema_names),
    }
    manifest_path = args.out_dir / "model_manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    print("[6/6] Done", flush=True)
    print(f"      {frontend_path} ({frontend_path.stat().st_size / 1024**2:.1f} MiB)")
    print(f"      {backend_path} ({backend_path.stat().st_size / 1024**2:.1f} MiB)")
    print(f"      {manifest_path}")


if __name__ == "__main__":
    main()
