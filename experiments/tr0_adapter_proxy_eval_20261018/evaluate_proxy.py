#!/usr/bin/env python3
"""Fresh-CV single-step x0 proxy evaluation for scaled TR-0 adapters."""
from __future__ import annotations

import csv
import hashlib
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torchaudio

ROOT = Path("/data/coding/sgmse")
CKPT = ROOT / "joint_train_logs/sb61i0gj_RRSD/epoch=118-step=80000.ckpt"
MANIFEST = ROOT / "experiments/tr0_fresh_cv_manifest_20261018/manifest.csv"
ADAPTER_ROOT = ROOT / "experiments/tr0_adapter_screen_scaled_20261018"
OUT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def load_rows():
    with MANIFEST.open(newline="") as f:
        return list(csv.DictReader(f))


def crop_pair(clean_path: str, noisy_path: str, target_len: int = 255 * 128):
    clean, sr1 = torchaudio.load(clean_path)
    noisy, sr2 = torchaudio.load(noisy_path)
    if sr1 != 16000 or sr2 != 16000:
        raise RuntimeError(f"sample rate mismatch {sr1}/{sr2}")
    clean, noisy = clean[:1], noisy[:1]
    n = min(clean.shape[-1], noisy.shape[-1])
    clean, noisy = clean[..., :n], noisy[..., :n]
    if n >= target_len:
        start = (n - target_len) // 2
        clean, noisy = clean[..., start:start + target_len], noisy[..., start:start + target_len]
    else:
        pad = target_len - n
        left = pad // 2
        clean = torch.nn.functional.pad(clean, (left, pad - left))
        noisy = torch.nn.functional.pad(noisy, (left, pad - left))
    scale = noisy.abs().amax().clamp_min(1e-12)
    return clean / scale, noisy / scale


def bootstrap_cluster(rows, key, reps=2000, seed=20261018):
    rng = np.random.default_rng(seed)
    speakers = sorted({r["speaker"] for r in rows})
    by = {s: [r for r in rows if r["speaker"] == s] for s in speakers}
    vals = []
    for _ in range(reps):
        chosen = rng.choice(speakers, size=len(speakers), replace=True)
        sample = [r for s in chosen for r in by[s]]
        vals.append(float(np.mean([r[key] for r in sample])))
    return [float(np.quantile(vals, 0.025)), float(np.quantile(vals, 0.975))]


def main():
    torch.set_num_threads(1)
    torch.cuda.manual_seed_all(20261018)
    from sgmse.joint_model import JointFrontendSgmseModel

    rows = load_rows()
    model = JointFrontendSgmseModel.load_from_checkpoint(
        CKPT, weights_only=False, map_location="cpu", backend_adapter_enable=True,
        backend_adapter_module=33, num_workers=0, batch_size=1, num_eval_files=0,
        frontend_trainable=False,
    ).to("cuda").eval()
    model.frontend.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    base_adapter = model.dnn.backend_adapter
    states = {
        "score_only_seed20261018": torch.load(ADAPTER_ROOT / "score_only/seed_20261018/adapter_state.pt", map_location="cuda", weights_only=True),
        "trust_region_seed20261018": torch.load(ADAPTER_ROOT / "trust_region/seed_20261018/adapter_state.pt", map_location="cuda", weights_only=True),
        "score_only_seed20261019": torch.load(ADAPTER_ROOT / "score_only/seed_20261019/adapter_state.pt", map_location="cuda", weights_only=True),
        "trust_region_seed20261019": torch.load(ADAPTER_ROOT / "trust_region/seed_20261019/adapter_state.pt", map_location="cuda", weights_only=True),
    }
    arms = ["baseline"] + list(states)
    out_rows = []
    with torch.inference_mode():
        for i, row in enumerate(rows):
            clean, noisy = crop_pair(row["clean"], row["noisy"])
            clean, noisy = clean.cuda(), noisy.cuda()
            X = model._sgmse_stft(clean)
            Yn = model._sgmse_stft(noisy)
            Yp = model.frontend.forward_transformed_spec(Yn)
            torch.manual_seed(20261018 + i)
            t = torch.rand(1, device="cuda") * (model.sde.T - model.t_eps) + model.t_eps
            mean, std = model.sde.marginal_prob(X, Yp, t)
            z = torch.randn_like(X)
            xt = mean + std[:, None, None, None] * z
            model.dnn.backend_adapter = None
            base_out = model(xt, Yp, t, y_cond=Yn)
            base_x0 = model._get_enhanced_spec(base_out, xt, t, mean, X, Yp)
            base_mse = float((base_x0 - X).abs().square().mean())
            scores = {"baseline": base_mse}
            for name, state in states.items():
                model.dnn.backend_adapter = base_adapter
                base_adapter.load_state_dict(state, strict=True)
                out = model(xt, Yp, t, y_cond=Yn)
                x0 = model._get_enhanced_spec(out, xt, t, mean, X, Yp)
                mse = float((x0 - X).abs().square().mean())
                scores[name] = mse
            model.dnn.backend_adapter = base_adapter
            for name, mse in scores.items():
                out_rows.append({"uttid": row["uttid"], "speaker": row["speaker"], "arm": name, "mse": mse, "delta_vs_baseline": mse - base_mse})

    (OUT / "per_utterance.csv").write_text("uttid,speaker,arm,mse,delta_vs_baseline\n" + "\n".join(
        f"{r['uttid']},{r['speaker']},{r['arm']},{r['mse']:.12g},{r['delta_vs_baseline']:.12g}" for r in out_rows
    ) + "\n")
    baseline = {r["uttid"]: r["mse"] for r in out_rows if r["arm"] == "baseline"}
    metrics = {"experiment_id": "tr0_adapter_proxy_eval_20261018", "status": "completed_fresh_cv_proxy_only", "manifest_sha256": sha256(MANIFEST), "n_utterances": len(rows), "n_speakers": len({r['speaker'] for r in rows}), "crop_samples": 255 * 128, "seed": 20261018, "arms": {}}
    for arm in arms:
        vals = [r for r in out_rows if r["arm"] == arm]
        deltas = [r["delta_vs_baseline"] for r in vals]
        rec = {"mean_mse": float(np.mean([r["mse"] for r in vals])), "mean_delta_vs_baseline": float(np.mean(deltas)), "speaker_cluster_ci_delta": bootstrap_cluster(vals, "delta_vs_baseline") if arm != "baseline" else [0.0, 0.0]}
        metrics["arms"][arm] = rec
    metrics["interpretation"] = "single-step clean x0 proxy only; no waveform inference or DNSMOS/PESQ/STOI/SI-SDR conclusion"
    (OUT / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
