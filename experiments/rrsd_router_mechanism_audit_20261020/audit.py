#!/usr/bin/env python
from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import torch


ROOT = Path("/data/coding/sgmse")
sys.path.insert(0, str(ROOT))
CKPT = ROOT / "joint_train_logs/sb61i0gj_RRSD/epoch=118-step=80000.ckpt"
SOURCE = ROOT / "sgmse/joint_model.py"
OUT = ROOT / "experiments/rrsd_router_mechanism_audit_20261020"


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    ckpt = torch.load(CKPT, map_location="cpu", weights_only=False)
    hp = ckpt["hyper_parameters"]
    state = ckpt["state_dict"]
    router = {k: v for k, v in state.items() if k.startswith("rrsd_router.")}
    result = {
        "experiment_id": "rrsd_router_mechanism_audit_20261020",
        "checkpoint_sha256": sha(CKPT),
        "joint_model_sha256": sha(SOURCE),
        "git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "global_step": int(ckpt["global_step"]),
        "hyper_parameters": {
            k: hp.get(k)
            for k in [
                "sde", "N", "rrsd_teacher", "rrsd_freeze_base", "rrsd_weight",
                "rrsd_future_weight", "rrsd_smooth_weight", "rrsd_future_gate_cost",
                "rrsd_use_gate", "contrastive_weight", "rrsd_perceptual_conflict_weight",
            ]
        },
        "state_dict": {
            "total_tensor_keys": len(state),
            "router_tensor_keys": len(router),
            "router_parameter_numel": int(sum(v.numel() for v in router.values())),
            "non_router_parameter_numel": int(sum(v.numel() for k, v in state.items() if not k.startswith("rrsd_router."))),
            "router_keys": sorted(router),
        },
        "source_locations": {
            "future_grad_function": "sgmse/joint_model.py:_rrsd_future_grad_loss (around lines 3791-4012)",
            "router_configuration": "sgmse/joint_model.py:parser and initialization (around lines 516-1128)",
        },
        "checks": {
            "future_grad_teacher": hp.get("rrsd_teacher") == "future_grad",
            "frozen_base": hp.get("rrsd_freeze_base") is True,
            "contrastive_disabled": float(hp.get("contrastive_weight", 0.0)) == 0.0,
            "explicit_gate_disabled": hp.get("rrsd_use_gate") is False,
            "router_only_state_keys_are_eight": len(router) == 8,
        },
        "dataset_modified": False,
        "checkpoint_modified": False,
        "training_started": False,
    }
    assert all(result["checks"].values()), result["checks"]
    (OUT / "metrics.json").write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
