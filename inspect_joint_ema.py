#!/usr/bin/env python3
"""
Inspect EMA correctness for a jointly-trained StoRM frontend + SGMSE backend checkpoint.

Preferred mode (recommended): pass --project-root.  The script reconstructs the exact
backend/frontend modules from checkpoint hyperparameters and compares checkpoint EMA
shadow tensors against the real named_parameters() order.  This avoids relying on Adam
exp_avg/exp_avg_sq, which may legitimately be missing for parameters that never received
a gradient.

It also scans sgmse/model.py for inconsistent EMA parameter collections, e.g. creating
EMA from all model parameters but updating/copying only self.dnn.parameters().
"""

from __future__ import annotations

import argparse
import csv
import math
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import torch
import torch.nn as nn


def load_ckpt(path: Path):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def rms(t: torch.Tensor) -> float:
    x = t.detach().float()
    if x.numel() == 0:
        return 0.0
    return float(torch.sqrt(torch.mean(x * x)).item())


def cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    x = a.detach().float().reshape(-1)
    y = b.detach().float().reshape(-1)
    denom = torch.linalg.vector_norm(x) * torch.linalg.vector_norm(y)
    if float(denom) == 0.0:
        return float("nan")
    return float(torch.dot(x, y).div(denom).item())


def group_name(name: str) -> str:
    if name.startswith("dnn."):
        return "dnn"
    if name.startswith("frontend."):
        return "frontend"
    return name.split(".", 1)[0]


def _split_prefixed_state(state: Mapping[str, Any], prefix: str) -> Dict[str, torch.Tensor]:
    return {
        k[len(prefix):]: v
        for k, v in state.items()
        if k.startswith(prefix) and isinstance(v, torch.Tensor)
    }


def _load_module_state(module: nn.Module, state: Mapping[str, torch.Tensor], label: str) -> None:
    result = module.load_state_dict(state, strict=True)
    if result.missing_keys or result.unexpected_keys:
        raise RuntimeError(
            f"{label} state mismatch: missing={result.missing_keys}, unexpected={result.unexpected_keys}"
        )


def reconstruct_named_parameters(
    project_root: Path, ckpt: Mapping[str, Any]
) -> List[Tuple[str, nn.Parameter]]:
    """Rebuild exactly the two modules used by export_joint_onnx.py."""
    root = project_root.resolve()
    if not (root / "sgmse").is_dir():
        raise FileNotFoundError(f"sgmse package not found under {root}")
    sys.path.insert(0, str(root))

    from sgmse.backbones import BackboneRegistry
    from sgmse.frontends.storm import StormFrontend

    hp = ckpt.get("hyper_parameters")
    state = ckpt.get("state_dict")
    if not isinstance(hp, Mapping) or not isinstance(state, Mapping):
        raise RuntimeError("checkpoint must contain hyper_parameters and state_dict")

    backbone_name = hp.get("backbone")
    if not isinstance(backbone_name, str):
        raise RuntimeError(f"invalid checkpoint backbone={backbone_name!r}")

    dnn_cls = BackboneRegistry.get_by_name(backbone_name)
    dnn_kwargs = dict(hp)
    if hp.get("train_type") == "mode2":
        # Match the exporter behavior for this checkpoint.
        dnn_kwargs["mode2_include_noisy_main"] = bool(hp.get("mode2_include_noisy_main", True))
    dnn = dnn_cls(**dnn_kwargs)
    frontend = StormFrontend(**dict(hp))

    _load_module_state(dnn, _split_prefixed_state(state, "dnn."), "dnn")
    _load_module_state(frontend, _split_prefixed_state(state, "frontend."), "frontend")

    tracked: List[Tuple[str, nn.Parameter]] = [
        (f"dnn.{name}", p) for name, p in dnn.named_parameters() if p.requires_grad
    ]
    if bool(hp.get("frontend_trainable", False)):
        tracked.extend(
            (f"frontend.{name}", p)
            for name, p in frontend.named_parameters()
            if p.requires_grad
        )
    return tracked


def optimizer_ids_and_missing_states(ckpt: Mapping[str, Any]):
    """Return optimizer param ids plus which ids lack Adam moment tensors."""
    opts = ckpt.get("optimizer_states")
    if not isinstance(opts, (list, tuple)) or not opts:
        return [], [], {}
    opt = opts[0]
    groups = opt.get("param_groups", []) if isinstance(opt, Mapping) else []
    state = opt.get("state", {}) if isinstance(opt, Mapping) else {}
    ids: List[int] = []
    for g in groups:
        if isinstance(g, Mapping):
            ids.extend(int(x) for x in g.get("params", []))

    missing: List[int] = []
    shapes: Dict[int, Tuple[int, ...]] = {}
    for pid in ids:
        st = state.get(pid, state.get(str(pid), {})) if isinstance(state, Mapping) else {}
        ref = None
        if isinstance(st, Mapping):
            for key in ("exp_avg", "exp_avg_sq", "max_exp_avg_sq"):
                if isinstance(st.get(key), torch.Tensor):
                    ref = st[key]
                    break
        if isinstance(ref, torch.Tensor):
            shapes[pid] = tuple(ref.shape)
        else:
            missing.append(pid)
    return ids, missing, shapes


def fallback_map_from_state_dict(
    state_dict: Mapping[str, Any], shadows: Sequence[torch.Tensor], optimizer_ids: Sequence[int]
):
    """Fallback only if model reconstruction fails.

    Uses EMA tensor shapes for optimizer entries whose Adam state is absent.  This is
    sufficient for diagnostics, but less trustworthy than reconstruct_named_parameters().
    """
    ids, missing, known_shapes = optimizer_ids_and_missing_states({
        "optimizer_states": []
    })
    del ids, missing, known_shapes  # silence static tools; real values supplied below

    items = [(k, v) for k, v in state_dict.items() if isinstance(v, torch.Tensor)]
    mapped: List[Tuple[str, torch.Tensor]] = []
    skipped: List[Tuple[str, Tuple[int, ...]]] = []
    j = 0
    for pos, pid in enumerate(optimizer_ids):
        if pos >= len(shadows) or not isinstance(shadows[pos], torch.Tensor):
            raise RuntimeError(f"EMA shadow[{pos}] missing/non-tensor")
        shape = tuple(shadows[pos].shape)
        while j < len(items) and tuple(items[j][1].shape) != shape:
            skipped.append((items[j][0], tuple(items[j][1].shape)))
            j += 1
        if j >= len(items):
            raise RuntimeError(
                f"fallback mapping failed at optimizer position={pos}, pid={pid}, EMA shape={shape}"
            )
        mapped.append((items[j][0], items[j][1]))
        j += 1
    while j < len(items):
        skipped.append((items[j][0], tuple(items[j][1].shape)))
        j += 1
    return mapped, skipped


def weighted_group_stats(rows: List[Dict[str, Any]]) -> Dict[str, Dict[str, float]]:
    groups: Dict[str, List[Dict[str, Any]]] = {}
    for r in rows:
        groups.setdefault(r["group"], []).append(r)
    out: Dict[str, Dict[str, float]] = {}
    for g, rs in groups.items():
        total_n = sum(int(r["numel"]) for r in rs)
        raw_e = sum((float(r["raw_rms"]) ** 2) * int(r["numel"]) for r in rs)
        ema_e = sum((float(r["ema_rms"]) ** 2) * int(r["numel"]) for r in rs)
        delta_e = sum((float(r["delta_rms"]) ** 2) * int(r["numel"]) for r in rs)
        raw_all = math.sqrt(raw_e / max(total_n, 1))
        ema_all = math.sqrt(ema_e / max(total_n, 1))
        delta_all = math.sqrt(delta_e / max(total_n, 1))
        out[g] = {
            "parameter_tensors": float(len(rs)),
            "numel": float(total_n),
            "raw_rms": raw_all,
            "ema_rms": ema_all,
            "delta_rms": delta_all,
            "relative_delta": delta_all / max(raw_all, 1e-12),
            "max_abs_delta": max(float(r["max_abs_delta"]) for r in rs),
            "mean_cosine": sum(float(r["cosine_raw_ema"]) for r in rs if math.isfinite(float(r["cosine_raw_ema"]))) /
                max(1, sum(math.isfinite(float(r["cosine_raw_ema"])) for r in rs)),
        }
    return out


def scan_model_py(project_root: Path) -> None:
    p = project_root / "sgmse" / "model.py"
    print("\n" + "=" * 96)
    print("Static EMA source scan")
    print("=" * 96)
    if not p.is_file():
        print(f"model.py not found: {p}")
        return

    text = p.read_text(encoding="utf-8", errors="replace")
    lines = text.splitlines()
    needles = (
        "ExponentialMovingAverage",
        ".ema.update",
        ".ema.store",
        ".ema.copy_to",
        ".ema.restore",
        "ema.update",
        "ema.store",
        "ema.copy_to",
        "ema.restore",
    )
    hits = [(i, line.rstrip()) for i, line in enumerate(lines, 1) if any(n in line for n in needles)]
    if hits:
        for i, line in hits:
            print(f"{i:5d}: {line}")
    else:
        print("No obvious EMA calls found in sgmse/model.py")

    compact = re.sub(r"\s+", "", text)
    init_all = "ExponentialMovingAverage(self.parameters()" in compact
    init_dnn = "ExponentialMovingAverage(self.dnn.parameters()" in compact
    update_dnn = "ema.update(self.dnn.parameters())" in compact
    store_dnn = "ema.store(self.dnn.parameters())" in compact
    copy_dnn = "ema.copy_to(self.dnn.parameters())" in compact
    restore_dnn = "ema.restore(self.dnn.parameters())" in compact
    update_all = "ema.update(self.parameters())" in compact
    store_all = "ema.store(self.parameters())" in compact
    copy_all = "ema.copy_to(self.parameters())" in compact
    restore_all = "ema.restore(self.parameters())" in compact

    print("\nStatic diagnosis:")
    if init_all and (update_dnn or store_dnn or copy_dnn or restore_dnn):
        print("  [HIGH-RISK] EMA is constructed from self.parameters(), but at least one later")
        print("              update/store/copy/restore call uses only self.dnn.parameters().")
        print("              With frontend_trainable=True these are different collections.")
    elif init_dnn and (update_all or store_all or copy_all or restore_all):
        print("  [HIGH-RISK] EMA is constructed from self.dnn.parameters(), but a later EMA call")
        print("              uses self.parameters().  Parameter count/order can differ.")
    else:
        print("  No simple all-parameters <-> dnn-only mismatch pattern was detected automatically.")
        print("  Verify manually that init/update/store/copy_to/restore always receive the SAME")
        print("  parameter collection in the SAME order.")

    print("\nImportant:")
    print("  Joint training alone does NOT guarantee correct EMA.  EMA correctness depends on")
    print("  the exact parameter list used when the shadows are created, updated, copied, and restored.")


def main() -> None:
    ap = argparse.ArgumentParser(description="Inspect joint StoRM+SGMSE checkpoint EMA")
    ap.add_argument("--ckpt", type=Path, required=True)
    ap.add_argument(
        "--project-root",
        type=Path,
        default=None,
        help="Project root containing sgmse/. Strongly recommended for exact named-parameter mapping.",
    )
    ap.add_argument("--csv", type=Path, default=Path("ema_diff.csv"))
    args = ap.parse_args()

    ckpt = load_ckpt(args.ckpt)
    state_dict = ckpt.get("state_dict")
    ema = ckpt.get("ema")
    hp = ckpt.get("hyper_parameters", {})
    if not isinstance(state_dict, Mapping):
        raise RuntimeError("checkpoint['state_dict'] missing")
    if not isinstance(ema, Mapping):
        raise RuntimeError("checkpoint['ema'] missing")
    shadows = ema.get("shadow_params")
    if not isinstance(shadows, (list, tuple)):
        raise RuntimeError("checkpoint['ema']['shadow_params'] missing/invalid")
    if not all(isinstance(x, torch.Tensor) for x in shadows):
        raise RuntimeError("checkpoint EMA contains non-tensor shadow entries")

    optimizer_ids, missing_opt_state, opt_shapes = optimizer_ids_and_missing_states(ckpt)

    print("=" * 96)
    print(f"checkpoint          : {args.ckpt}")
    print(f"global_step         : {ckpt.get('global_step')}")
    print(f"EMA decay           : {ema.get('decay')}")
    print(f"EMA num_updates     : {ema.get('num_updates')}")
    print(f"frontend_trainable  : {hp.get('frontend_trainable') if isinstance(hp, Mapping) else None}")
    print(f"train_type          : {hp.get('train_type') if isinstance(hp, Mapping) else None}")
    print(f"state_dict tensors  : {sum(isinstance(v, torch.Tensor) for v in state_dict.values())}")
    print(f"optimizer param ids : {len(optimizer_ids)}")
    print(f"EMA shadow params   : {len(shadows)}")
    print(f"optimizer states with Adam moments : {len(opt_shapes)}")
    print(f"optimizer params WITHOUT Adam moments: {len(missing_opt_state)}")
    if missing_opt_state:
        preview = ", ".join(str(x) for x in missing_opt_state[:30])
        suffix = " ..." if len(missing_opt_state) > 30 else ""
        print(f"  missing optimizer state ids: {preview}{suffix}")
        print("  [INFO] This is legal: optimizer state is created lazily when a parameter receives a gradient.")

    # Exact mapping by rebuilding the architecture whenever possible.
    named_params: List[Tuple[str, torch.Tensor]]
    mapping_mode = ""
    if args.project_root is not None:
        try:
            rebuilt = reconstruct_named_parameters(args.project_root, ckpt)
            named_params = [(name, p.detach().cpu()) for name, p in rebuilt]
            mapping_mode = "exact model reconstruction / named_parameters()"
        except Exception as exc:
            print(f"\n[WARNING] exact model reconstruction failed: {type(exc).__name__}: {exc}")
            print("          Falling back to checkpoint-order/EMA-shape mapping; this is less reliable.")
            if not optimizer_ids:
                raise RuntimeError("cannot fall back: checkpoint has no optimizer param ids") from exc
            named_params, skipped = fallback_map_from_state_dict(state_dict, shadows, optimizer_ids)
            mapping_mode = "fallback state_dict order + EMA shapes"
            if skipped:
                print(f"          fallback skipped {len(skipped)} state_dict tensors (likely buffers)")
    else:
        if not optimizer_ids:
            raise RuntimeError("pass --project-root for exact mapping; optimizer state is unavailable for fallback")
        named_params, skipped = fallback_map_from_state_dict(state_dict, shadows, optimizer_ids)
        mapping_mode = "fallback state_dict order + EMA shapes"
        if skipped:
            print(f"fallback skipped {len(skipped)} state_dict tensors (likely buffers)")

    print(f"\nEMA mapping mode: {mapping_mode}")
    print(f"mapped named parameters: {len(named_params)}")

    if len(named_params) != len(shadows):
        raise RuntimeError(
            f"EMA count mismatch: named parameters={len(named_params)} vs shadows={len(shadows)}. "
            "Do NOT zip/apply this EMA blindly."
        )
    if optimizer_ids and len(optimizer_ids) != len(named_params):
        print(
            f"[WARNING] optimizer param count={len(optimizer_ids)} differs from named parameter count={len(named_params)}"
        )

    missing_pos = set()
    if optimizer_ids and missing_opt_state:
        pid_to_pos = {pid: i for i, pid in enumerate(optimizer_ids)}
        missing_pos = {pid_to_pos[pid] for pid in missing_opt_state if pid in pid_to_pos}

    rows: List[Dict[str, Any]] = []
    shape_bad = []
    for idx, ((name, raw), shadow) in enumerate(zip(named_params, shadows)):
        raw = raw.detach().cpu()
        shadow = shadow.detach().cpu()
        if tuple(raw.shape) != tuple(shadow.shape):
            shape_bad.append((idx, name, tuple(raw.shape), tuple(shadow.shape)))
            continue
        rawf = raw.float()
        emaf = shadow.float()
        delta = emaf - rawf
        rr = rms(rawf)
        er = rms(emaf)
        dr = rms(delta)
        optimizer_id = optimizer_ids[idx] if idx < len(optimizer_ids) else None
        rows.append({
            "index": idx,
            "optimizer_id": optimizer_id,
            "optimizer_state_present": idx not in missing_pos if optimizer_ids else "unknown",
            "group": group_name(name),
            "name": name,
            "shape": str(tuple(raw.shape)),
            "numel": int(raw.numel()),
            "raw_rms": rr,
            "ema_rms": er,
            "delta_rms": dr,
            "relative_delta": dr / max(rr, 1e-12),
            "max_abs_delta": float(delta.abs().max().item()) if delta.numel() else 0.0,
            "cosine_raw_ema": cosine(rawf, emaf),
            "raw_finite": bool(torch.isfinite(rawf).all().item()),
            "ema_finite": bool(torch.isfinite(emaf).all().item()),
        })

    if shape_bad:
        print("\n[ERROR] EMA shadow shapes do not match exact named-parameter order:")
        for item in shape_bad[:30]:
            print("  ", item)
        raise RuntimeError(
            f"{len(shape_bad)} EMA tensors mismatch parameter shapes. This strongly indicates wrong EMA collection/order."
        )
    else:
        print("[OK] every EMA shadow shape matches the corresponding named parameter shape")

    group_counts: Dict[str, int] = {}
    for r in rows:
        group_counts[r["group"]] = group_counts.get(r["group"], 0) + 1
    print("\nEMA parameter ranges:")
    for g, n in group_counts.items():
        inds = [int(r["index"]) for r in rows if r["group"] == g]
        print(f"  {g:12s}: {n:4d} tensors, EMA indices {min(inds)}..{max(inds)}")

    if missing_pos:
        print("\nParameters lacking optimizer Adam moments:")
        for r in rows:
            if int(r["index"]) in missing_pos:
                print(f"  [{r['index']:3d}] pid={r['optimizer_id']} {r['name']} {r['shape']}")

    summary = weighted_group_stats(rows)
    print("\nRaw vs EMA numerical difference by group:")
    for g, s in summary.items():
        print(
            f"  {g:12s}: tensors={int(s['parameter_tensors']):4d}, numel={int(s['numel']):12d}, "
            f"raw_rms={s['raw_rms']:.6g}, ema_rms={s['ema_rms']:.6g}, "
            f"delta_rms={s['delta_rms']:.6g}, relative_delta={s['relative_delta']:.6g}, "
            f"mean_cos={s['mean_cosine']:.6g}, max_abs_delta={s['max_abs_delta']:.6g}"
        )

    # Heuristic specifically useful for a trainable frontend.
    if "dnn" in summary and "frontend" in summary:
        drel = summary["dnn"]["relative_delta"]
        frel = summary["frontend"]["relative_delta"]
        ratio = frel / max(drel, 1e-12)
        print(f"\nfrontend/dnn EMA relative-delta ratio: {ratio:.6g}x")
        if ratio > 5.0 and frel > 1e-3:
            print("  [SUSPICIOUS] frontend EMA is much farther from raw weights than dnn EMA.")
            print("               Possible causes: frontend shadows were stale/not updated, or wrong apply/update semantics.")
        elif ratio < 0.2 and drel > 1e-3:
            print("  [SUSPICIOUS] dnn EMA is much farther from raw weights than frontend EMA.")
        else:
            print("  No extreme frontend-vs-dnn EMA divergence by this simple metric.")

    nonfinite = [r for r in rows if not r["raw_finite"] or not r["ema_finite"]]
    if nonfinite:
        print(f"\n[ERROR] {len(nonfinite)} parameter tensors contain NaN/Inf")
    else:
        print("\n[OK] all raw and EMA tensors are finite")

    worst = sorted(rows, key=lambda r: float(r["relative_delta"]), reverse=True)[:20]
    print("\nLargest relative raw -> EMA changes:")
    for r in worst:
        print(
            f"  [{r['index']:3d}] rel={r['relative_delta']:.6g} cos={r['cosine_raw_ema']:.6g} "
            f"maxΔ={r['max_abs_delta']:.6g}  {r['name']} {r['shape']}"
        )

    args.csv.parent.mkdir(parents=True, exist_ok=True)
    if rows:
        with args.csv.open("w", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        print(f"\nCSV written: {args.csv}")

    if args.project_root is not None:
        scan_model_py(args.project_root)

    print("\n" + "=" * 96)
    print("How to interpret this checkpoint")
    print("=" * 96)
    print("1) If shape mapping fails: EMA parameter collection/order is definitely inconsistent.")
    print("2) If shapes match but frontend raw-vs-EMA delta is enormous relative to dnn:")
    print("   frontend EMA may have been initialized but not updated correctly.")
    print("3) If shapes and deltas look reasonable but use-EMA inference fails:")
    print("   inspect EMA copy/store/restore at inference time; it may copy to a different subset/order.")
    print("4) Missing Adam state for some params only means those params did not receive optimizer state;")
    print("   it is NOT by itself proof of a broken EMA. The parameter names printed above show where they are.")


if __name__ == "__main__":
    main()
