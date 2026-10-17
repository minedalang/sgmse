#!/usr/bin/env python3
"""One-utterance rollout-state diagnostic for the fixed StoRM checkpoint."""
from __future__ import annotations
import argparse, csv, hashlib, json, time
from pathlib import Path
import numpy as np
import soundfile as sf
import torch

import sys

ROOT = Path(__file__).resolve().parents[2]
EXP = Path(__file__).resolve().parent
CKPT = ROOT / "joint_train_logs/sb61i0gj_RRSD/epoch=118-step=80000.ckpt"
MANIFEST = ROOT / "experiments/storm_error_attribution_p7_20261017/baseline_manifest.csv"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EXPECTED_CKPT_SHA = "ab50510ca7fd80798b70f3f0b5365b17448a5f98c75d38755b82212ae2a119c9"
EXPECTED_MANIFEST_SHA = "77b64db9da6a1ff372a22dcda4a6316743133f383103aaffb3dbce2c671c178e"
sys.path.insert(0, str(ROOT))

def sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""): h.update(block)
    return h.hexdigest()

def match_length(x, ref):
    n, m = x.shape[-1], ref.shape[-1]
    if n == m: return x
    if n > m:
        s = (n - m) // 2
        return x[..., s:s+m]
    p = m - n
    return torch.nn.functional.pad(x, (p // 2, p - p // 2))

def finite(x):
    return bool(torch.isfinite(x.real if torch.is_complex(x) else x).all())

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--row-index", type=int, default=0)
    parser.add_argument("--output", type=Path, default=EXP / "metrics.json")
    args = parser.parse_args()
    if sha256(CKPT) != EXPECTED_CKPT_SHA: raise RuntimeError("checkpoint hash changed")
    if sha256(MANIFEST) != EXPECTED_MANIFEST_SHA: raise RuntimeError("manifest hash changed")
    with MANIFEST.open(newline="", encoding="utf-8") as f: rows = list(csv.DictReader(f))
    row = rows[args.row_index]
    noisy_path, clean_path = Path(row["noisy_path"]), Path(row["clean_path"])
    for path, key in ((noisy_path, "noisy_sha256"), (clean_path, "clean_sha256")):
        if sha256(path) != row[key]: raise RuntimeError(f"input hash changed: {path}")
    from sgmse.joint_data_module import JointWaveDataModule
    from sgmse.joint_model import JointSemambaSgmseModel
    from sgmse.util.other import pad_spec
    t0 = time.perf_counter()
    model = JointSemambaSgmseModel.load_from_checkpoint(
        str(CKPT), map_location="cpu", strict=True, data_module_cls=JointWaveDataModule,
        base_dir=".", batch_size=1, num_workers=0, dummy=True, weights_only=False)
    model.eval(no_ema=False).to(DEVICE)
    noisy_np, noisy_sr = sf.read(noisy_path, dtype="float32", always_2d=False)
    clean_np, clean_sr = sf.read(clean_path, dtype="float32", always_2d=False)
    if noisy_sr != 16000 or clean_sr != 16000: raise RuntimeError("expected 16 kHz inputs")
    noisy = torch.from_numpy(np.asarray(noisy_np, dtype=np.float32)).to(DEVICE)[None,None,:]
    clean = torch.from_numpy(np.asarray(clean_np, dtype=np.float32)).to(DEVICE)[None,None,:]
    clean = match_length(clean, noisy)
    norm = noisy.abs().amax(dim=-1, keepdim=True).clamp_min(1e-12)
    with torch.inference_mode():
        y_pre = match_length(model._frontend_forward(noisy / norm) * norm, noisy)
        y_main = pad_spec(torch.unsqueeze(model._forward_transform(model._stft(y_pre.squeeze(1) / norm.squeeze(1))), 0))
        y_cond = pad_spec(torch.unsqueeze(model._forward_transform(model._stft(noisy.squeeze(1) / norm.squeeze(1))), 0))
        x_clean = pad_spec(torch.unsqueeze(model._forward_transform(model._stft(clean.squeeze(1) / norm.squeeze(1))), 0))
    if not all(finite(x) for x in (y_main, y_cond, x_clean)): raise FloatingPointError("nonfinite spectrum")
    traces = []
    def sink(item):
        traces.append({"step_index": int(item["step_index"]), "t": float(item["t"].flatten()[0].item()), "x_t": item["x_t"].detach().cpu()})
    seed = 20261018
    torch.manual_seed(seed)
    if DEVICE.type == "cuda": torch.cuda.manual_seed(seed)
    with torch.inference_mode():
        model.enhance(y_pre.squeeze(0), y_cond=noisy.squeeze(0), sampler_type="pc", predictor="reverse_diffusion", corrector="none", N=30, snr=0.5, trace_sink=sink)
    if len(traces) != int(model.sde.N): raise RuntimeError(f"expected {model.sde.N} traces, got {len(traces)}")
    stats = []
    with torch.inference_mode():
        for item in traces:
            t = torch.tensor([item["t"]], device=DEVICE, dtype=torch.float32)
            mean, std = model.sde.marginal_prob(x_clean, y_main, t)
            state = item["x_t"].to(DEVICE)
            scale = std[:,None,None,None].clamp_min(1e-12)
            z_clean = (state - mean) / scale
            z_prior = (state - y_main) / scale
            power_clean = torch.abs(z_clean).float().square()
            power_prior = torch.abs(z_prior).float().square()
            stats.append({"step_index": item["step_index"], "t": item["t"], "standardized_abs2_mean": float(power_clean.mean()), "standardized_abs2_median": float(power_clean.median()), "standardized_abs2_p95": float(torch.quantile(power_clean.flatten(), 0.95)), "prior_centered_abs2_mean": float(power_prior.mean()), "state_finite": finite(item["x_t"])})
    values = np.asarray([s["standardized_abs2_mean"] for s in stats], dtype=np.float64)
    prior_values = np.asarray([s["prior_centered_abs2_mean"] for s in stats], dtype=np.float64)
    payload = {"experiment_id":"corrective_forcing_rollout_smoke_20261018", "status":"completed", "device":str(DEVICE), "checkpoint_sha256":sha256(CKPT), "manifest_sha256":sha256(MANIFEST), "filename":row["filename"], "uttid":row["uttid"], "seed":seed, "sampler":"PC reverse_diffusion, N=30, corrector=none, snr=0.5", "n_trace_states":len(stats), "expected_complex_standardized_abs2":2.0, "runtime_seconds":time.perf_counter()-t0, "summary":{"clean_center_mean_over_steps":float(values.mean()), "prior_center_mean_over_steps":float(prior_values.mean()), "clean_center_median_over_steps":float(np.median(values)), "first_step":stats[0], "last_step":stats[-1], "min_clean_center_step_mean":float(values.min()), "max_clean_center_step_mean":float(values.max())}, "states":stats, "interpretation":"Offline clean-aware rollout diagnostic only. The prior-centered control is included because OUVE prior_sampling is centered at Y_pre while the training marginal includes exp(-theta*T)*X_clean; residual deviation is not by itself proof of harmful train-inference mismatch or evidence that Corrective Forcing will improve quality."}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload["summary"], indent=2))

if __name__ == "__main__": main()
