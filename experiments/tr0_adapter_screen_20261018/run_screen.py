#!/usr/bin/env python3
"""TR-0 short adapter screen: score-only vs baseline-relative trust region.

This is deliberately a small, adapter-only training screen. The frontend,
Router, and all original NCSN++ parameters stay frozen. Only adapter state
dicts are saved; the original checkpoint is never written.
"""
from __future__ import annotations

import gc
import hashlib
import json
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path("/data/coding/sgmse")
CKPT = ROOT / "joint_train_logs/sb61i0gj_RRSD/epoch=118-step=80000.ckpt"
DATA_ROOT = Path("/data/datasets/dns_ins20/dns_snr_-20_-5/synthetic_snr-20_-5")
TRAIN_SCP = DATA_ROOT / "scp_new/tr_synthetic/wav.scp"
VALID_SCP = DATA_ROOT / "scp_new/cv_synthetic/wav.scp"
CLEAN_DIR = DATA_ROOT / "clean"
OUT = Path(os.environ.get("TR0_SCREEN_OUT", str(Path(__file__).resolve().parent)))
sys.path.insert(0, str(ROOT))

STEPS = 200
SEEDS = (20261018, 20261019)
ARMS = ("score_only", "trust_region")
TRUST_WEIGHT = float(os.environ.get("TR0_TRUST_WEIGHT", "1.0"))
TRUST_MARGIN = 0.0


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def make_model(seed: int):
    from sgmse.joint_model import JointFrontendSgmseModel

    model = JointFrontendSgmseModel.load_from_checkpoint(
        CKPT,
        weights_only=False,
        map_location="cpu",
        backend_adapter_enable=True,
        backend_adapter_module=33,
        train_wav_scp=str(TRAIN_SCP),
        valid_wav_scp=str(VALID_SCP),
        clean_dir=str(CLEAN_DIR),
        num_workers=0,
        batch_size=1,
        num_eval_files=0,
        frontend_trainable=False,
        no_wandb=True,
    )
    model = model.to("cuda")
    model.train()
    model.frontend.eval()
    model.dnn.eval()
    model.dnn.backend_adapter.train()
    for name, param in model.named_parameters():
        param.requires_grad_(name.startswith("dnn.backend_adapter."))
    # The frozen backbone has no stochastic layers relevant to this config;
    # keeping it in eval mode makes the comparison deterministic.
    return model


def one_step(model, batch, arm: str, optimizer, step_idx: int):
    x_wav, y_wav, _spk = [v.cuda(non_blocking=True) for v in batch]
    X = model._sgmse_stft(x_wav)
    Y_noisy = model._sgmse_stft(y_wav)
    with torch.no_grad():
        Y_pre = model.frontend.forward_transformed_spec(Y_noisy)

    details = model._step((X, Y_noisy, Y_pre), step_idx, return_details=True)
    base_loss = details["base_loss"]
    x0_adapt = details["x_hat_spec"]
    with torch.no_grad():
        adapter = model.dnn.backend_adapter
        model.dnn.backend_adapter = None
        base_out = model(details["x_t"], details["y"], details["t"], y_cond=details["y_cond"])
        model.dnn.backend_adapter = adapter
        x0_base = model._get_enhanced_spec(
            base_out, details["x_t"], details["t"], details["mean"], details["x"], details["y"]
        )
        base_err = (x0_base - X).abs().square().detach()

    adapted_err = (x0_adapt - X).abs().square()
    trust_loss = torch.relu(adapted_err - base_err + TRUST_MARGIN).mean()
    weighted_trust = TRUST_WEIGHT * trust_loss if arm == "trust_region" else trust_loss.new_zeros(())
    total_loss = base_loss + weighted_trust
    optimizer.zero_grad(set_to_none=True)
    total_loss.backward()
    torch.nn.utils.clip_grad_norm_(model.dnn.backend_adapter.parameters(), max_norm=1.0)
    optimizer.step()
    with torch.no_grad():
        adapter_norm = torch.sqrt(sum(p.square().sum() for p in model.dnn.backend_adapter.parameters()))
    return {
        "base_loss": float(base_loss.detach()),
        "trust_loss": float(trust_loss.detach()),
        "weighted_trust": float(weighted_trust.detach()),
        "total_loss": float(total_loss.detach()),
        "base_x0_mse": float(base_err.mean()),
        "adapt_x0_mse": float(adapted_err.detach().mean()),
        "adapter_norm": float(adapter_norm),
    }


def run_arm(arm: str, seed: int) -> dict:
    seed_all(seed)
    model = make_model(seed)
    model.data_module.setup("fit")
    loader = model.data_module.train_dataloader()
    iterator = iter(loader)
    optimizer = torch.optim.AdamW(model.dnn.backend_adapter.parameters(), lr=1e-4, weight_decay=0.0)
    history = []
    t0 = time.perf_counter()
    for step in range(STEPS):
        try:
            batch = next(iterator)
        except StopIteration:
            iterator = iter(loader)
            batch = next(iterator)
        row = one_step(model, batch, arm, optimizer, step)
        row["step"] = step + 1
        history.append(row)
    elapsed = time.perf_counter() - t0
    arm_dir = OUT / arm / f"seed_{seed}"
    arm_dir.mkdir(parents=True, exist_ok=True)
    torch.save({k: v.detach().cpu() for k, v in model.dnn.backend_adapter.state_dict().items()}, arm_dir / "adapter_state.pt")
    summary = {
        "arm": arm,
        "seed": seed,
        "steps": STEPS,
        "elapsed_seconds": elapsed,
        "steps_per_second": STEPS / elapsed,
        "adapter_parameters": sum(p.numel() for p in model.dnn.backend_adapter.parameters()),
        "initial": history[0],
        "final": history[-1],
        "mean_last_20": {k: float(np.mean([r[k] for r in history[-20:]])) for k in history[-1] if k != "step"},
        "history": history,
    }
    (arm_dir / "metrics.json").write_text(json.dumps(summary, indent=2) + "\n")
    del model, loader, iterator, optimizer
    gc.collect()
    torch.cuda.empty_cache()
    return summary


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("TR-0 screen requires CUDA")
    OUT.mkdir(parents=True, exist_ok=True)
    summaries = []
    for arm in ARMS:
        for seed in SEEDS:
            print(f"[TR-0] starting arm={arm} seed={seed}", flush=True)
            summaries.append(run_arm(arm, seed))
            print(f"[TR-0] finished arm={arm} seed={seed}", flush=True)
    result = {
        "experiment_id": "tr0_adapter_screen_20261018",
        "status": "completed_short_screen_no_inference_evaluation",
        "checkpoint": {"path": str(CKPT), "sha256": sha256(CKPT)},
        "dataset": {"train_scp": str(TRAIN_SCP), "valid_scp": str(VALID_SCP), "modified": False},
        "protocol": {"arms": list(ARMS), "seeds": list(SEEDS), "steps": STEPS, "trust_weight": TRUST_WEIGHT, "trust_margin": TRUST_MARGIN, "adapter_module": 33, "batch_size": 1, "device": "cuda"},
        "summaries": [{k: v for k, v in s.items() if k not in ("history",)} for s in summaries],
        "note": "Training proxy only. Fresh-CV waveform inference and DNSMOS/PESQ/STOI/SI-SDR evaluation are required before any scientific conclusion.",
    }
    (OUT / "metrics.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
