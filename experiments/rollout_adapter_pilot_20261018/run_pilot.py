#!/usr/bin/env python3
"""Bounded rollout-state adapter pilot.

The adapter is zero initialized and is the only trainable component. States are
cached from the frozen PC rollout on a fresh-CV manifest, then the adapter is
trained to reduce clean x0 NMSE on those rollout states. This is a proxy pilot,
not a final enhancement evaluation.
"""
from __future__ import annotations
import argparse, csv, hashlib, json, sys, time
from pathlib import Path
import numpy as np
import soundfile as sf
import torch

ROOT = Path(__file__).resolve().parents[2]
EXP = Path(__file__).resolve().parent
CKPT = ROOT / "joint_train_logs/sb61i0gj_RRSD/epoch=118-step=80000.ckpt"
MANIFEST = ROOT / "experiments/rollout_prediction_gap_20261018/fresh_cv_manifest.csv"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CKPT_SHA = "ab50510ca7fd80798b70f3f0b5365b17448a5f98c75d38755b82212ae2a119c9"
SELECTED_STEPS = (3, 5, 7, 9)
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

def load_model():
    from sgmse.joint_data_module import JointWaveDataModule
    from sgmse.joint_model import JointSemambaSgmseModel
    model = JointSemambaSgmseModel.load_from_checkpoint(
        str(CKPT), map_location="cpu", strict=False,
        data_module_cls=JointWaveDataModule, base_dir=".", batch_size=1,
        num_workers=0, dummy=True, weights_only=False,
        backend_adapter_enable=True, backend_adapter_module=33,
    )
    if model.dnn.backend_adapter is None:
        raise RuntimeError("backend adapter was not constructed")
    model.eval(no_ema=False).to(DEVICE)
    return model

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

def x0_nmse(model, state, y_main, y_cond, x_clean, t):
    score = model(state, y_main, t, y_cond=y_cond)
    x0 = model._rrsd_score_to_x0(state, y_main, t, score)[0]
    num = torch.abs(x0 - x_clean).float().square().mean()
    den = torch.abs(x_clean).float().square().mean().clamp_min(1e-12)
    return num / den

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--cache", type=Path, default=EXP / "rollout_state_cache.pt")
    args = ap.parse_args()
    if sha256(CKPT) != CKPT_SHA: raise RuntimeError("checkpoint hash changed")
    if not MANIFEST.is_file(): raise FileNotFoundError(MANIFEST)
    with MANIFEST.open(newline="", encoding="utf-8") as f: rows = list(csv.DictReader(f))
    if len(rows) < 8: raise RuntimeError("fresh CV manifest must contain 8 rows")
    from sgmse.util.other import pad_spec
    start = time.perf_counter()
    model = load_model()
    cache_items = []
    for idx, row in enumerate(rows[:8]):
        pre, noisy, y_main, y_cond, x_clean = prepare(model, row, pad_spec)
        traces = []
        def sink(item):
            if int(item["step_index"]) in SELECTED_STEPS:
                traces.append({"step_index": int(item["step_index"]), "t": float(item["t"].flatten()[0].item()), "x_t": item["x_t"].detach().cpu()})
        seed = 20261018 + idx
        torch.manual_seed(seed)
        if DEVICE.type == "cuda": torch.cuda.manual_seed(seed)
        with torch.inference_mode():
            model.enhance(pre.squeeze(0), y_cond=noisy.squeeze(0), sampler_type="pc", predictor="reverse_diffusion", corrector="none", N=30, snr=0.5, trace_sink=sink)
        if len(traces) != len(SELECTED_STEPS): raise RuntimeError(f"missing selected states for {row['filename']}")
        for tr in traces:
            t = torch.tensor([tr["t"]], dtype=torch.float32)
            cache_items.append({"utt_index": idx, "filename": row["filename"], "step_index": tr["step_index"], "t": t, "x_t": tr["x_t"], "y_main": y_main.detach().cpu(), "y_cond": y_cond.detach().cpu(), "x_clean": x_clean.detach().cpu()})
    args.cache.parent.mkdir(parents=True, exist_ok=True)
    torch.save(cache_items, args.cache)

    # Utterance split: 6 train, 2 holdout. Keep the frozen score and frontend deterministic.
    for p in model.parameters(): p.requires_grad = False
    adapter = model.dnn.backend_adapter
    for p in adapter.parameters(): p.requires_grad = True
    model.dnn.backend_adapter.train()
    optimizer = torch.optim.Adam(adapter.parameters(), lr=1e-3)
    train_items = [x for x in cache_items if x["utt_index"] < 6]
    hold_items = [x for x in cache_items if x["utt_index"] >= 6]
    def evaluate(items):
        vals=[]
        model.eval()
        with torch.inference_mode():
            for item in items:
                vals.append(float(x0_nmse(model, item["x_t"].to(DEVICE), item["y_main"].to(DEVICE), item["y_cond"].to(DEVICE), item["x_clean"].to(DEVICE), item["t"].to(DEVICE))))
        model.dnn.backend_adapter.train()
        return float(np.mean(vals)), vals
    baseline_train, _ = evaluate(train_items)
    baseline_hold, hold_before = evaluate(hold_items)
    history=[]
    generator = torch.Generator().manual_seed(20261018)
    for step in range(args.steps):
        idx = int(torch.randint(len(train_items), (1,), generator=generator).item())
        item = train_items[idx]
        optimizer.zero_grad(set_to_none=True)
        loss = x0_nmse(model, item["x_t"].to(DEVICE), item["y_main"].to(DEVICE), item["y_cond"].to(DEVICE), item["x_clean"].to(DEVICE), item["t"].to(DEVICE))
        if not torch.isfinite(loss): raise FloatingPointError("nonfinite adapter loss")
        loss.backward(); optimizer.step()
        if step in (0, 9, 49, 99, args.steps-1): history.append({"step":step+1,"train_loss":float(loss.detach())})
    trained_train, _ = evaluate(train_items)
    trained_hold, hold_after = evaluate(hold_items)
    torch.save({"adapter_state_dict":adapter.state_dict(), "base_checkpoint":str(CKPT), "selected_steps":SELECTED_STEPS, "train_steps":args.steps}, EXP / "adapter_state.pt")
    payload={"experiment_id":"rollout_adapter_pilot_20261018","status":"completed","device":str(DEVICE),"checkpoint_sha256":sha256(CKPT),"manifest":str(MANIFEST),"n_utterances":8,"selected_steps":SELECTED_STEPS,"train_utterances":6,"holdout_utterances":2,"train_steps":args.steps,"adapter_parameters":sum(p.numel() for p in adapter.parameters()),"baseline":{"train_nmse":baseline_train,"holdout_nmse":baseline_hold},"trained":{"train_nmse":trained_train,"holdout_nmse":trained_hold},"holdout_before":hold_before,"holdout_after":hold_after,"holdout_relative_change":(trained_hold-baseline_hold)/max(baseline_hold,1e-12),"history":history,"runtime_seconds":time.perf_counter()-start,"interpretation":"Proxy only: the adapter is trained on cached rollout states from six fresh-CV utterances and evaluated on two held-out utterances. No final audio metrics are included; a holdout proxy gain is required before any audio evaluation, and a negative or unstable result stops this route."}
    (EXP / "metrics.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k:payload[k] for k in ('baseline','trained','holdout_relative_change','adapter_parameters')}, indent=2))

if __name__ == "__main__": main()
