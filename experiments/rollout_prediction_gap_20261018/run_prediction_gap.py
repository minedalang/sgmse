#!/usr/bin/env python3
"""Compare x0 prediction errors on analytic forward states and reverse rollouts."""
from __future__ import annotations
import argparse, csv, hashlib, json, sys, time
from pathlib import Path
import numpy as np
import soundfile as sf
import torch

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
CKPT = ROOT / "joint_train_logs/sb61i0gj_RRSD/epoch=118-step=80000.ckpt"
DEFAULT_MANIFEST = ROOT / "experiments/storm_error_attribution_p7_20261017/baseline_manifest.csv"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CKPT_SHA = "ab50510ca7fd80798b70f3f0b5365b17448a5f98c75d38755b82212ae2a119c9"
MANIFEST_SHA = "77b64db9da6a1ff372a22dcda4a6316743133f383103aaffb3dbce2c671c178e"
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

def prepare(model, row, pad_spec):
    noisy_np, nsr = sf.read(Path(row["noisy_path"]), dtype="float32", always_2d=False)
    clean_np, csr = sf.read(Path(row["clean_path"]), dtype="float32", always_2d=False)
    if nsr != 16000 or csr != 16000: raise RuntimeError("expected 16 kHz")
    noisy = torch.from_numpy(np.asarray(noisy_np, dtype=np.float32)).to(DEVICE)[None,None,:]
    clean = torch.from_numpy(np.asarray(clean_np, dtype=np.float32)).to(DEVICE)[None,None,:]
    clean = match_length(clean, noisy)
    norm = noisy.abs().amax(dim=-1, keepdim=True).clamp_min(1e-12)
    with torch.inference_mode():
        pre = match_length(model._frontend_forward(noisy / norm) * norm, noisy)
        y_main = pad_spec(torch.unsqueeze(model._forward_transform(model._stft(pre.squeeze(1) / norm.squeeze(1))), 0))
        y_cond = pad_spec(torch.unsqueeze(model._forward_transform(model._stft(noisy.squeeze(1) / norm.squeeze(1))), 0))
        x_clean = pad_spec(torch.unsqueeze(model._forward_transform(model._stft(clean.squeeze(1) / norm.squeeze(1))), 0))
    return pre, noisy, y_main, y_cond, x_clean

def nmse(xhat, target):
    num = torch.abs(xhat - target).float().square().mean()
    den = torch.abs(target).float().square().mean().clamp_min(1e-12)
    return float((num / den).item())

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-utterances", type=int, default=4)
    ap.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    ap.add_argument("--output", type=Path, default=OUT / "metrics.json")
    args = ap.parse_args()
    if sha256(CKPT) != CKPT_SHA: raise RuntimeError("checkpoint hash changed")
    manifest_path = args.manifest.resolve()
    if manifest_path == DEFAULT_MANIFEST and sha256(manifest_path) != MANIFEST_SHA: raise RuntimeError("manifest hash changed")
    with manifest_path.open(newline="", encoding="utf-8") as f: rows = list(csv.DictReader(f))[:args.n_utterances]
    from sgmse.joint_data_module import JointWaveDataModule
    from sgmse.joint_model import JointSemambaSgmseModel
    from sgmse.util.other import pad_spec
    start = time.perf_counter()
    model = JointSemambaSgmseModel.load_from_checkpoint(str(CKPT), map_location="cpu", strict=True, data_module_cls=JointWaveDataModule, base_dir=".", batch_size=1, num_workers=0, dummy=True, weights_only=False)
    model.eval(no_ema=False).to(DEVICE)
    utterance_results = []
    for idx, row in enumerate(rows):
        pre, noisy, y_main, y_cond, x_clean = prepare(model, row, pad_spec)
        traces = []
        def sink(item):
            traces.append({"step_index": int(item["step_index"]), "t": float(item["t"].flatten()[0].item()), "x_t": item["x_t"].detach().clone()})
        seed = 20261018 + idx
        torch.manual_seed(seed)
        if DEVICE.type == "cuda": torch.cuda.manual_seed(seed)
        with torch.inference_mode():
            model.enhance(pre.squeeze(0), y_cond=noisy.squeeze(0), sampler_type="pc", predictor="reverse_diffusion", corrector="none", N=30, snr=0.5, trace_sink=sink)
        if len(traces) != 30: raise RuntimeError(f"expected 30 states, got {len(traces)}")
        steps = []
        with torch.inference_mode():
            for step in traces:
                t = torch.tensor([step["t"],], device=DEVICE, dtype=torch.float32)
                state = step["x_t"].to(DEVICE)
                mean, std = model.sde.marginal_prob(x_clean, y_main, t)
                torch.manual_seed(seed * 1000 + step["step_index"])
                z = torch.randn_like(x_clean)
                analytic = mean + std[:,None,None,None] * z
                score_roll = model(state, y_main, t, y_cond=y_cond)
                score_analytic = model(analytic, y_main, t, y_cond=y_cond)
                x0_roll = model._rrsd_score_to_x0(state, y_main, t, score_roll)[0]
                x0_analytic = model._rrsd_score_to_x0(analytic, y_main, t, score_analytic)[0]
                roll_err = nmse(x0_roll, x_clean)
                fwd_err = nmse(x0_analytic, x_clean)
                steps.append({"step_index": step["step_index"], "t": step["t"], "rollout_x0_nmse": roll_err, "analytic_forward_x0_nmse": fwd_err, "delta_rollout_minus_forward": roll_err - fwd_err, "finite": finite(x0_roll) and finite(x0_analytic)})
        roll = np.asarray([s["rollout_x0_nmse"] for s in steps])
        fwd = np.asarray([s["analytic_forward_x0_nmse"] for s in steps])
        utterance_results.append({"filename": row["filename"], "uttid": row["uttid"], "seed": seed, "rollout_x0_nmse_mean": float(roll.mean()), "analytic_forward_x0_nmse_mean": float(fwd.mean()), "delta_mean": float((roll-fwd).mean()), "rollout_last_step": float(roll[-1]), "analytic_last_step": float(fwd[-1]), "steps": steps})
        print(row["filename"], utterance_results[-1]["delta_mean"], flush=True)
    deltas = np.asarray([r["delta_mean"] for r in utterance_results])
    payload = {"experiment_id":"rollout_prediction_gap_20261018", "status":"completed", "device":str(DEVICE), "checkpoint_sha256":sha256(CKPT), "manifest_path":str(manifest_path), "manifest_sha256":sha256(manifest_path), "n_utterances":len(utterance_results), "sampler":"PC reverse_diffusion, N=30, corrector=none, snr=0.5", "runtime_seconds":time.perf_counter()-start, "aggregate":{"rollout_x0_nmse_mean":float(np.mean([r["rollout_x0_nmse_mean"] for r in utterance_results])), "analytic_forward_x0_nmse_mean":float(np.mean([r["analytic_forward_x0_nmse_mean"] for r in utterance_results])), "delta_mean":float(deltas.mean()), "delta_median":float(np.median(deltas)), "delta_positive_utterances":int((deltas>0).sum())}, "utterances":utterance_results, "interpretation":"This compares x0 prediction errors on reverse rollout states with errors on analytically sampled forward states at matched times. It is a diagnostic of off-path prediction behavior, not a causal proof that Corrective Forcing will improve final enhancement quality."}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload["aggregate"], indent=2))

if __name__ == "__main__": main()
