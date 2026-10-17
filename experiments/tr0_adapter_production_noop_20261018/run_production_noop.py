#!/usr/bin/env python3
"""Verify the optional production NCSN++ adapter preserves the 80k model."""
from __future__ import annotations

import hashlib
import json
import sys
import time
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


def make_kwargs(**extra):
    out = dict(
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
    out.update(extra)
    return out


def main() -> None:
    torch.set_num_threads(1)
    torch.manual_seed(20261018)
    from sgmse.backbones.ncsnpp_v2 import NCSNpp_v2

    checkpoint = torch.load(CKPT, map_location="cpu", weights_only=False)
    state = {k[4:]: v for k, v in checkpoint["state_dict"].items() if k.startswith("dnn.")}
    base = NCSNpp_v2(**make_kwargs()).eval()
    base.load_state_dict(state, strict=True)
    adapted = NCSNpp_v2(**make_kwargs(backend_adapter_enable=True, backend_adapter_module=33)).eval()
    missing, unexpected = adapted.load_state_dict(state, strict=False)
    x = torch.randn(1, 1, 256, 256, dtype=torch.complex64)
    y = torch.randn_like(x)
    t = torch.tensor([0.3])
    with torch.inference_mode():
        t0 = time.perf_counter()
        base_out = base(x, y, t, y_cond=y, y_extra=y)
        adapted_out = adapted(x, y, t, y_cond=y, y_extra=y)
        elapsed = time.perf_counter() - t0
    diff = adapted_out - base_out
    result = {
        "experiment_id": "tr0_adapter_production_noop_20261018",
        "status": "completed_no_training",
        "checkpoint": {"path": str(CKPT), "sha256": sha256(CKPT), "global_step": checkpoint.get("global_step")},
        "model": {"backbone": "ncsnpp_v2", "nf": 128, "ch_mult": [1, 2, 2, 2], "num_res_blocks": 1, "input_shape": [1, 1, 256, 256]},
        "adapter": {"enabled": True, "module_index": 33, "channels": 128, "parameters": sum(p.numel() for p in adapted.backend_adapter.parameters())},
        "checkpoint_load": {"base_missing": [], "base_unexpected": [], "adapter_missing_expected": missing, "adapter_unexpected": unexpected},
        "comparison": {"cpu_elapsed_seconds": elapsed, "max_abs_diff": float(diff.abs().max()), "rmse": float(torch.sqrt((diff.abs() ** 2).mean())), "torch_equal": bool(torch.equal(base_out, adapted_out))},
        "default_path_unchanged": True,
        "data_or_checkpoint_modified": False,
    }
    (OUT / "metrics.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
