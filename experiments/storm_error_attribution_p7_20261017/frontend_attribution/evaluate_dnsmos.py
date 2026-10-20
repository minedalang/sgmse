#!/usr/bin/env python3
"""Run paired DNSMOS on existing P7-C outputs using the mymetrics env."""
from __future__ import annotations

import csv
import argparse
import hashlib
import importlib
import json
import time
import warnings
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from torchmetrics.functional.audio.dnsmos import deep_noise_suppression_mean_opinion_score

PROJECT = Path(__file__).resolve().parents[3]
EXP = PROJECT / "experiments/storm_error_attribution_p7_20261017"
MANIFEST = EXP / "baseline_manifest.csv"
FRONT = EXP / "frontend_only"
OFF = PROJECT / "experiment_logs/rrsd_tcut_screen/off"
ROUTER = PROJECT / "experiment_logs/rrsd_tcut_screen/router"
OUT = Path(__file__).resolve().parent
SR = 16000
REPS = 10000
SEED = 20261019
_DEVICE_CACHE: dict[str, torch.device] = {}
_RUNTIME_INFO: dict[str, object] = {}


def load(path: Path) -> np.ndarray:
    x, sr = sf.read(path, dtype="float32", always_2d=False)
    if x.ndim == 2:
        x = x.mean(axis=-1)
    if sr != SR or x.ndim != 1:
        raise ValueError(f"Invalid audio {path}: sr={sr}, shape={x.shape}")
    if not np.isfinite(x).all():
        raise ValueError(f"Nonfinite values in {path}")
    peak = float(np.max(np.abs(x))) + 1e-12
    if peak > 1.0:
        x = x / peak
    return np.ascontiguousarray(x, dtype=np.float32)


def _session_providers(device: torch.device) -> dict[str, list[str]]:
    """Use TorchMetrics' cached loader and report the providers on its sessions."""
    module = importlib.import_module("torchmetrics.functional.audio.dnsmos")
    loader = getattr(module, "_cached_load_session", None)
    if loader is None:
        raise RuntimeError("TorchMetrics does not expose its cached DNSMOS session loader")
    paths = {
        "sig_bak_ovr": f"{module.DNSMOS_DIR}/DNSMOS/sig_bak_ovr.onnx",
        "model_v8": f"{module.DNSMOS_DIR}/DNSMOS/model_v8.onnx",
    }
    return {
        name: loader(path, device, 1).get_providers()
        for name, path in paths.items()
    }


def resolve_device(requested: str = "auto") -> torch.device:
    """Prefer CUDA only when Torch and both DNSMOS ONNX sessions support it."""
    requested = str(requested or "auto").strip().lower()
    if requested in _DEVICE_CACHE:
        return _DEVICE_CACHE[requested]
    if requested == "auto":
        target = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    else:
        try:
            target = torch.device(requested)
        except (TypeError, RuntimeError) as exc:
            raise ValueError("--device must be 'auto', 'cpu', or 'cuda[:index]'.") from exc
    if target.type not in {"cpu", "cuda"}:
        raise ValueError("--device must be 'auto', 'cpu', or 'cuda[:index]'.")

    if target.type == "cuda":
        try:
            if not torch.cuda.is_available():
                raise RuntimeError("PyTorch CUDA is unavailable")
            index = target.index if target.index is not None else torch.cuda.current_device()
            if index < 0 or index >= torch.cuda.device_count():
                raise RuntimeError(f"CUDA device index {index} is unavailable")
            target = torch.device("cuda", index)
            import onnxruntime as ort
            available = ort.get_available_providers()
            if "CUDAExecutionProvider" not in available:
                raise RuntimeError(f"ONNX Runtime providers do not include CUDA: {available}")
            providers = _session_providers(target)
            missing = [name for name, active in providers.items()
                       if "CUDAExecutionProvider" not in active]
            if missing:
                raise RuntimeError(f"DNSMOS sessions lack CUDAExecutionProvider: {missing}")
        except Exception as exc:
            warnings.warn(f"DNSMOS CUDA unavailable; falling back to CPU: {exc}", RuntimeWarning)
            target = torch.device("cpu")

    if target.type == "cpu":
        providers = _session_providers(target)
    _RUNTIME_INFO.clear()
    _RUNTIME_INFO.update({
        "device": str(target),
        "execution_provider": "CUDAExecutionProvider"
        if all("CUDAExecutionProvider" in p for p in providers.values())
        else "CPUExecutionProvider",
        "onnx_session_providers": providers,
    })
    _DEVICE_CACHE[requested] = target
    return target


def mos(x: np.ndarray, device: str = "auto") -> np.ndarray:
    selected = resolve_device(device)

    def _compute(target: torch.device) -> torch.Tensor:
        return deep_noise_suppression_mean_opinion_score(
            torch.from_numpy(x), fs=SR, personalized=False, device=str(target),
            cache_session=True, num_threads=1,
        )

    with torch.no_grad():
        try:
            result = _compute(selected)
        except Exception as exc:
            if selected.type != "cuda":
                raise
            warnings.warn(f"DNSMOS CUDA inference failed; retrying on CPU: {exc}", RuntimeWarning)
            _DEVICE_CACHE[str(device or "auto").strip().lower()] = torch.device("cpu")
            selected = resolve_device("cpu")
            result = _compute(selected)
    return result.detach().cpu().numpy().reshape(4)


def ci(x: np.ndarray) -> list[float]:
    rng = np.random.default_rng(SEED)
    ids = rng.integers(0, len(x), size=(REPS, len(x)))
    means = x[ids].mean(axis=1)
    return [float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))]


def summarize(x: np.ndarray) -> dict[str, float]:
    return {"mean": float(x.mean()), "std": float(x.std(ddof=1)),
            "median": float(np.median(x))}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--device", default="auto",
        help="DNSMOS device: auto (CUDA when available), cpu, or cuda[:index].",
    )
    args = parser.parse_args()

    started = time.time()
    resolve_device(args.device)
    with MANIFEST.open(newline="", encoding="utf-8") as f:
        entries = list(csv.DictReader(f))
    if len(entries) != 32:
        raise ValueError(f"Expected 32 entries, got {len(entries)}")
    labels = ("p808", "sig", "bak", "ovrl")
    rows = []
    for i, item in enumerate(entries, 1):
        row = {"filename": item["filename"], "uttid": item["uttid"]}
        for arm, path in (
            ("frontend", FRONT / item["filename"]),
            ("off", OFF / item["filename"]),
            ("router", ROUTER / item["filename"]),
        ):
            values = mos(load(path), device=args.device)
            for name, value in zip(labels, values):
                row[f"{arm}_{name}"] = float(value)
        rows.append(row)
        if i % 4 == 0:
            print(f"evaluated utterance {i}/{len(entries)}", flush=True)

    paired = {}
    per_arm = {}
    for label in labels:
        arrays = {arm: np.asarray([r[f"{arm}_{label}"] for r in rows])
                  for arm in ("frontend", "off", "router")}
        per_arm[label] = {arm: summarize(v) for arm, v in arrays.items()}
        paired[label] = {}
        for a, b in (("frontend", "off"), ("router", "off"), ("router", "frontend")):
            delta = arrays[a] - arrays[b]
            paired[label][f"{a}_minus_{b}"] = {
                "mean_delta": float(delta.mean()),
                "median_delta": float(np.median(delta)),
                "bootstrap_95ci": ci(delta),
                "a_wins": int((delta > 0).sum()),
                "b_wins": int((delta < 0).sum()),
                "ties": int((delta == 0).sum()),
            }
    with (OUT / "dnsmos_per_utterance.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    report = {
        "experiment_id": "storm_error_attribution_p7_20261017_frontend_dnsmos",
        "n": len(rows),
        "sample_rate_hz": SR,
        "personalized": False,
        **_RUNTIME_INFO,
        "input_normalization": "same as mymetrics.calc_dnsmos: divide peak if >1",
        "manifest_sha256": hashlib.sha256(MANIFEST.read_bytes()).hexdigest(),
        "arms": {"frontend": "frontend_only", "off": str(OFF.relative_to(PROJECT)),
                 "router": str(ROUTER.relative_to(PROJECT))},
        "bootstrap": {"replicates": REPS, "seed": SEED,
                      "scope": "exploratory; previously screened development split"},
        "torchmetrics": __import__("torchmetrics").__version__,
        "metrics": {"per_arm": per_arm, "paired": paired},
        "runtime_seconds": time.time() - started,
        "training": False,
        "audio_written_or_modified": False,
    }
    (OUT / "dnsmos_metrics.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(report["metrics"], indent=2), flush=True)
    print(f"runtime_seconds={report['runtime_seconds']:.3f}", flush=True)


if __name__ == "__main__":
    main()
