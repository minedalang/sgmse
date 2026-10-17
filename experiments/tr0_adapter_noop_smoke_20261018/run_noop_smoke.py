#!/usr/bin/env python3
"""CPU exact-no-op smoke for a candidate output residual adapter.

The adapter is external to the checkpoint and is zero initialized. This smoke
does not train or write any checkpoint; it verifies the no-op contract and
records the parameter budget for a future implementation.
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
import time
from pathlib import Path

import torch
from torch import nn
from copy import deepcopy

ROOT = Path("/data/coding/sgmse")
CKPT = ROOT / "joint_train_logs/sb61i0gj_RRSD/epoch=118-step=80000.ckpt"
OUT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


class ZeroInitComplexResidualAdapter(nn.Module):
    """A minimal candidate: zero initialized residual 1x1 conv on real/imag."""

    def __init__(self, channels: int = 2):
        super().__init__()
        self.proj = nn.Conv2d(channels, channels, kernel_size=1, bias=True)
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        ri = torch.cat([z.real, z.imag], dim=1)
        d = self.proj(ri)
        return z + torch.complex(d[:, :1], d[:, 1:2])


class ZeroInitFeatureResidualAdapter(nn.Module):
    """Zero initialized residual adapter for a decoder feature map."""

    def __init__(self, channels: int):
        super().__init__()
        self.proj = nn.Conv2d(channels, channels, kernel_size=1, bias=True)
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return h + self.proj(h)


class DecoderResidualWrapper(nn.Module):
    def __init__(self, base: nn.Module, channels: int):
        super().__init__()
        self.base = base
        self.adapter = ZeroInitFeatureResidualAdapter(channels)

    def forward(self, *args, **kwargs):
        return self.adapter(self.base(*args, **kwargs))


def main() -> None:
    torch.set_num_threads(1)
    torch.manual_seed(20261018)
    from sgmse.backbones.ncsnpp_v2 import NCSNpp_v2

    hp = dict(
        nf=128,
        ch_mult=[1, 2, 2, 2],
        num_res_blocks=1,
        attn_resolutions=[0],
        cross_attn=False,
        film_modulate=False,
        film_levels="first",
        selector_type="none",
        mode2_include_noisy_main=True,
    )
    dnn = NCSNpp_v2(**hp).eval()
    checkpoint = torch.load(CKPT, map_location="cpu", weights_only=False)
    dnn.load_state_dict({k[4:]: v for k, v in checkpoint["state_dict"].items() if k.startswith("dnn.")})

    x = torch.randn(1, 1, 256, 256, dtype=torch.complex64)
    y = torch.randn_like(x)
    t = torch.tensor([0.3])
    with torch.inference_mode():
        t0 = time.perf_counter()
        base = dnn(x, y, t, y_cond=y, y_extra=y)
        base_seconds = time.perf_counter() - t0
        adapter = ZeroInitComplexResidualAdapter().eval()
        adapted = adapter(base)
        # Simulate a production insertion at the audited late decoder module 33
        # without changing the repository model or checkpoint.
        wrapped_dnn = deepcopy(dnn)
        decoder_idx = 33
        decoder_channels = int(wrapped_dnn.all_modules[decoder_idx].out_ch)
        wrapped_dnn.all_modules[decoder_idx] = DecoderResidualWrapper(
            wrapped_dnn.all_modules[decoder_idx], decoder_channels
        )
        wrapped = wrapped_dnn(x, y, t, y_cond=y, y_extra=y)
    diff = adapted - base
    decoder_diff = wrapped - base
    metrics = {
        "experiment_id": "tr0_adapter_noop_smoke_20261018",
        "status": "completed_no_training",
        "checkpoint": {"path": str(CKPT), "sha256": sha256(CKPT), "global_step": checkpoint.get("global_step")},
        "model": {"backbone": "ncsnpp_v2", "nf": 128, "ch_mult": [1, 2, 2, 2], "num_res_blocks": 1, "input_shape": [1, 1, 256, 256]},
        "adapter": {"type": "zero_init_complex_residual_1x1", "parameters": sum(p.numel() for p in adapter.parameters()), "trainable_in_smoke": False},
        "forward": {"base_seconds_cpu": base_seconds, "base_shape": list(base.shape), "base_dtype": str(base.dtype)},
        "no_op": {"max_abs_diff": float(diff.abs().max()), "rmse": float(torch.sqrt((diff.abs() ** 2).mean())), "allclose_atol_0": bool(torch.equal(adapted, base))},
        "decoder_no_op": {"module_index": decoder_idx, "channels": decoder_channels, "adapter_parameters": sum(p.numel() for p in wrapped_dnn.all_modules[decoder_idx].adapter.parameters()), "max_abs_diff": float(decoder_diff.abs().max()), "rmse": float(torch.sqrt((decoder_diff.abs() ** 2).mean())), "allclose_atol_0": bool(torch.equal(wrapped, base))},
        "data_or_checkpoint_modified": False,
    }
    (OUT / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
