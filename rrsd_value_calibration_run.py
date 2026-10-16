#!/usr/bin/env python3
"""Run the opt-in RRSD local-value calibration on a small paired subset.

This diagnostic writes JSONL traces only. It does not write enhanced audio or
the repository-level metrics CSV.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path


def seed_for(name, seed):
    h = hashlib.blake2b(name.encode(), digest_size=4).digest()
    return (int(seed) + int.from_bytes(h, "little")) % (2 ** 32)


def mono(path, sr):
    import librosa
    import soundfile as sf
    import numpy as np
    import torch
    x, got = sf.read(str(path), always_2d=False)
    if getattr(x, "ndim", 1) > 1:
        x = x.mean(axis=-1)
    if int(got) != int(sr):
        x = librosa.resample(x.astype("float32"), orig_sr=int(got), target_sr=int(sr))
    return torch.from_numpy(np.asarray(x)).float()


def match(x, ref):
    import torch.nn.functional as F
    if x.shape[-1] == ref.shape[-1]:
        return x
    if x.shape[-1] > ref.shape[-1]:
        d = x.shape[-1] - ref.shape[-1]
        return x[..., d // 2:d // 2 + ref.shape[-1]]
    d = ref.shape[-1] - x.shape[-1]
    return F.pad(x, (d // 2, d - d // 2))


def probe(model, tr, target, candidates, k, step_size):
    import torch
    import torch.nn.functional as F
    x, y_main, y_cond = tr["x_t"], tr["y_main"], tr["y_cond"]
    t0, score = tr["t"].flatten().float(), tr["score_base"]
    sigma, wx = tr["sigma"], tr["w_x"]
    base, front = tr["x0_base"], tr["x0_front"]
    alpha = tr["alpha_tile"].float()
    alpha = F.interpolate(alpha, size=base.shape[-2:], mode="bilinear", align_corners=False)
    dt0 = float(step_size.detach().flatten()[0].item() if torch.is_tensor(step_size) else step_size)
    errors = []
    for scale in candidates:
        routed = score + (wx * (float(scale) * alpha) * (front - base) /
                          sigma.square().clamp_min(1e-8)).to(score.dtype)
        dt = torch.minimum(torch.full_like(t0, max(dt0, 0.0)),
                           (t0 - float(model.t_eps)).clamp_min(0.0))
        cur, tc = model._rrsd_reverse_mean_step(x=x, y_main=y_main, t=t0, score=routed, dt=dt)
        for _ in range(1, int(k)):
            sc = model(cur, y_main, tc, y_cond=y_cond)
            dt = torch.minimum(torch.full_like(tc, max(dt0, 0.0)),
                               (tc - float(model.t_eps)).clamp_min(0.0))
            cur, tc = model._rrsd_reverse_mean_step(x=cur, y_main=y_main, t=tc, score=sc, dt=dt)
        sc = model(cur, y_main, tc, y_cond=y_cond)
        clean0, _, _, _ = model._rrsd_score_to_x0(cur, y_main, tc, sc)
        errors.append(float(torch.abs(clean0 - target).float().square().flatten(1).mean().item()))
    b = max(errors[0], 1e-12)
    a = alpha.detach().float()
    return {"candidates": list(candidates), "future_x0_mse": errors,
            "relative_gain": [(b - e) / b for e in errors],
            "best_candidate": float(candidates[min(range(len(errors)), key=errors.__getitem__)]),
            "alpha_mean": float(a.mean()), "alpha_max": float(a.max()),
            "alpha_shape": list(a.shape)}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--manifest", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--max-files", type=int, default=1)
    p.add_argument("--max-steps", type=int, default=2)
    p.add_argument("--N", type=int, default=8)
    p.add_argument("--K", type=int, default=4)
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--device", default="cuda")
    p.add_argument("--use-ema", action="store_true")
    args = p.parse_args()
    import torch
    from sgmse.joint_data_module import JointWaveDataModule
    from sgmse.joint_model import JointSemambaSgmseModel
    from sgmse.util.other import pad_spec
    rows = list(csv.DictReader(Path(args.manifest).open(newline="", encoding="utf-8")))[:args.max_files]
    out = Path(args.out); out.mkdir(parents=True, exist_ok=False)
    model = JointSemambaSgmseModel.load_from_checkpoint(
        args.ckpt, map_location="cpu", strict=True, data_module_cls=JointWaveDataModule,
        base_dir=".", batch_size=1, num_workers=0, dummy=True, weights_only=False)
    model.eval(no_ema=not args.use_ema).to(args.device)
    candidates = [0.0, 0.25, 0.5, 0.75, 1.0]
    with (out / "state_candidates.jsonl").open("w", encoding="utf-8") as fout:
        for row in rows:
            name = row["filename"]
            torch.manual_seed(seed_for(name, args.seed))
            noisy = mono(row["noisy"], 16000).to(args.device)[None, None, :]
            clean = mono(row["clean"], 16000).to(args.device)[None, :]
            norm = noisy.abs().amax(dim=-1, keepdim=True).clamp_min(1e-12)
            y_pre = match(model._frontend_forward(noisy / norm) * norm, noisy)
            clean = match(clean, noisy[0]).squeeze(0)
            # _stft([B,T]) returns [B,F,T]; the model's padded spectra use
            # [B,C,F,T], so add the same singleton channel axis as enhance().
            target_spec = model._forward_transform(model._stft(clean / norm.squeeze()))
            # clean is 1-D here, so STFT returns [F,T].  The score model
            # expects [B,C,F,T], matching enhance()'s two leading axes.
            target = pad_spec(target_spec.unsqueeze(0).unsqueeze(0))
            records = 0
            def sink(payload):
                nonlocal records
                if records >= args.max_steps or payload.get("calibrator_trace") is None:
                    return
                tr = dict(payload["calibrator_trace"]); tr["stepsize"] = payload["stepsize"]
                rec = probe(model, tr, target, candidates, args.K, payload["stepsize"])
                rec.update({"filename": name, "step": int(payload["step_index"]),
                            "t": float(payload["t"].flatten()[0].item())})
                fout.write(json.dumps(rec, sort_keys=True) + "\n"); fout.flush(); records += 1
            model.enhance(y_pre.squeeze(0), y_cond=noisy.squeeze(0), sampler_type="pc",
                          predictor="reverse_diffusion", corrector="none", N=args.N, snr=0.5,
                          rrsd=True, rrsd_scale=1.0, trace_sink=sink)
            print(f"{name}: wrote {records} states", flush=True)
    (out / "config.json").write_text(json.dumps(vars(args), indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
