#!/usr/bin/env python3
"""Verify adapter-only gradients with the locked NCSN++ checkpoint."""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import torch

ROOT = Path("/data/coding/sgmse")
CKPT = ROOT / "joint_train_logs/sb61i0gj_RRSD/epoch=118-step=80000.ckpt"
OUT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def main() -> None:
    torch.set_num_threads(1)
    torch.manual_seed(20261018)
    from sgmse.backbones.ncsnpp_v2 import NCSNpp_v2

    ckpt = torch.load(CKPT, map_location="cpu", weights_only=False)
    state = {k[4:]: v for k, v in ckpt["state_dict"].items() if k.startswith("dnn.")}
    model = NCSNpp_v2(
        nf=128, ch_mult=[1, 2, 2, 2], num_res_blocks=1,
        attn_resolutions=[0], cross_attn=False, film_modulate=False,
        film_levels="first", selector_type="none", mode2_include_noisy_main=True,
        backend_adapter_enable=True, backend_adapter_module=33,
    )
    model.load_state_dict(state, strict=False)
    for p in model.parameters():
        p.requires_grad_(False)
    for p in model.backend_adapter.parameters():
        p.requires_grad_(True)
    model.train()
    x = torch.randn(1, 1, 256, 256, dtype=torch.complex64)
    y = torch.randn_like(x)
    t = torch.tensor([0.3])
    target = torch.randn_like(x)
    out = model(x, y, t, y_cond=y, y_extra=y)
    loss = (out.real - target.real).square().mean() + (out.imag - target.imag).square().mean()
    loss.backward()
    adapter_grads = [p.grad for p in model.backend_adapter.parameters()]
    base_params = [p for n, p in model.named_parameters() if not n.startswith("backend_adapter.")]
    result = {
        "experiment_id": "tr0_adapter_gradient_smoke_20261018",
        "status": "completed_gradient_smoke_no_training",
        "checkpoint_sha256": sha256(CKPT),
        "adapter_module_index": 33,
        "adapter_parameters": sum(p.numel() for p in model.backend_adapter.parameters()),
        "loss": float(loss.detach()),
        "adapter_grad": {"all_present": all(g is not None for g in adapter_grads), "max_abs": max(float(g.abs().max()) for g in adapter_grads), "finite": all(bool(torch.isfinite(g).all()) for g in adapter_grads)},
        "frozen_base": {"all_requires_grad_false": all(not p.requires_grad for p in base_params), "all_grads_none": all(p.grad is None for p in base_params)},
        "dataset_modified": False,
        "checkpoint_modified": False,
    }
    (OUT / "metrics.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
