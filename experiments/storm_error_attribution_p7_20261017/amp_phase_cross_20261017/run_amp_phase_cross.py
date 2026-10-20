#!/usr/bin/env python3
"""Run the two fixed P7-C amplitude/phase swaps on the existing 32-file manifest.

P = STFT(frontend-only Y_pre); G = STFT(Router-off X_gen).
The only constructed signals are |P| angle(G) and |G| angle(P).
This is a zero-training, offline diagnostic; it does not search any parameters.
"""
from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
import math
import os
import platform
import sys
import time
import traceback
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf
import torch
from pesq import pesq
from pystoi import stoi

ROOT = Path(__file__).resolve().parents[3]
EXP = ROOT / "experiments/storm_error_attribution_p7_20261017"
OUT = Path(__file__).resolve().parent
MANIFEST = EXP / "baseline_manifest.csv"
FRONT = EXP / "frontend_only"
OFF = ROOT / "experiment_logs/rrsd_tcut_screen/off"
ROUTER = ROOT / "experiment_logs/rrsd_tcut_screen/router"
BASE_DIR = EXP / "frontend_attribution"
BASE_METRIC_CSVS = {
    "spectral_and_si_sdr": BASE_DIR / "per_utterance_metrics.csv",
    "pesq_stoi": BASE_DIR / "pesq_stoi_per_utterance.csv",
    "dnsmos": BASE_DIR / "dnsmos_per_utterance.csv",
}
DNS_EVALUATOR = BASE_DIR / "evaluate_dnsmos.py"
SR = 16000
N_FFT = 510
HOP = 128
BOOTSTRAP_REPS = 10_000
BOOTSTRAP_SEED = 20261021
ARM_DIRS = {
    "magP_phaseG": OUT / "magP_phaseG",
    "magG_phaseP": OUT / "magG_phaseP",
}
METRIC_NAMES = (
    "pesq_wb", "stoi", "si_sdr", "dnsmos_p808", "dnsmos_sig", "dnsmos_bak", "dnsmos_ovrl",
)
BASE_ARMS = ("frontend", "off", "router")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def write_json(path: Path, obj: Any) -> None:
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")


def write_config(config: dict[str, Any]) -> None:
    # JSON is a strict subset of YAML 1.2, so this remains parseable as config.yaml
    # without introducing a serializer dependency.
    write_json(OUT / "config.yaml", config)


def load_wave(path: Path) -> tuple[np.ndarray, int]:
    x, sr = sf.read(path, dtype="float32", always_2d=False)
    if x.ndim == 2:
        x = x.mean(axis=-1)
    if x.ndim != 1:
        raise ValueError(f"Expected mono audio after downmix; got {x.shape}: {path}")
    if sr != SR:
        raise ValueError(f"Expected {SR} Hz, got {sr}: {path}")
    if not np.isfinite(x).all():
        raise ValueError(f"Nonfinite samples: {path}")
    return np.ascontiguousarray(x, dtype=np.float32), int(sr)


def stft(x: np.ndarray) -> torch.Tensor:
    tensor = torch.from_numpy(np.ascontiguousarray(x, dtype=np.float32))
    window = torch.hann_window(N_FFT, periodic=True, dtype=torch.float32)
    return torch.stft(
        tensor,
        n_fft=N_FFT,
        hop_length=HOP,
        win_length=N_FFT,
        window=window,
        center=True,
        return_complex=True,
    )


def istft(spec: torch.Tensor, length: int) -> np.ndarray:
    window = torch.hann_window(N_FFT, periodic=True, dtype=torch.float32)
    out = torch.istft(
        spec,
        n_fft=N_FFT,
        hop_length=HOP,
        win_length=N_FFT,
        window=window,
        center=True,
        length=length,
    )
    y = out.detach().cpu().numpy().astype(np.float32, copy=False)
    if y.ndim != 1 or y.shape[0] != length:
        raise ValueError(f"iSTFT returned shape {y.shape}; expected ({length},)")
    if not np.isfinite(y).all():
        raise ValueError("iSTFT returned nonfinite waveform")
    return np.ascontiguousarray(y)


def waveform_reconstruction(original: np.ndarray, restored: np.ndarray) -> dict[str, float | int]:
    delta = restored.astype(np.float64) - original.astype(np.float64)
    rmse = float(np.sqrt(np.mean(delta * delta)))
    denom = float(np.sqrt(np.mean(original.astype(np.float64) ** 2))) + 1e-12
    return {
        "length_samples": int(restored.size),
        "rmse": rmse,
        "relative_rmse": rmse / denom,
        "max_abs_error": float(np.max(np.abs(delta))),
    }


def spectral_consistency(target: torch.Tensor, waveform: np.ndarray) -> dict[str, float]:
    projected = stft(waveform)
    if projected.shape != target.shape:
        raise ValueError(f"STFT consistency shape mismatch: {projected.shape} vs {target.shape}")
    diff = (projected - target).detach().cpu().numpy().astype(np.complex128)
    ref = target.detach().cpu().numpy().astype(np.complex128)
    rel = float(np.linalg.norm(diff.ravel()) / (np.linalg.norm(ref.ravel()) + 1e-18))
    return {
        "stft_projection_relative_l2": rel,
        "stft_projection_max_abs_error": float(np.max(np.abs(diff))),
    }


def si_sdr(reference: np.ndarray, estimate: np.ndarray) -> float:
    ref = reference.astype(np.float64) - float(np.mean(reference, dtype=np.float64))
    est = estimate.astype(np.float64) - float(np.mean(estimate, dtype=np.float64))
    denom = float(np.dot(ref, ref)) + 1e-12
    target = (float(np.dot(est, ref)) / denom) * ref
    noise = est - target
    return 10.0 * math.log10((float(np.dot(target, target)) + 1e-12) /
                             (float(np.dot(noise, noise)) + 1e-12))


def bootstrap_ci(values: np.ndarray, seed: int) -> list[float]:
    values = np.asarray(values, dtype=np.float64)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(values), size=(BOOTSTRAP_REPS, len(values)))
    means = values[idx].mean(axis=1)
    return [float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))]


def describe(values: np.ndarray) -> dict[str, float]:
    values = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(values.mean()),
        "std": float(values.std(ddof=1)),
        "median": float(np.median(values)),
    }


def metric_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    summaries: dict[str, Any] = {"n": len(rows), "arms": {}, "paired_new_minus_existing": {}}
    for arm in (*ARM_DIRS.keys(), *BASE_ARMS):
        summaries["arms"][arm] = {}
        for metric in METRIC_NAMES:
            vals = np.asarray([float(row[f"{arm}_{metric}"]) for row in rows], dtype=np.float64)
            summaries["arms"][arm][metric] = describe(vals)
    ci_seed = BOOTSTRAP_SEED
    for new_arm in ARM_DIRS:
        summaries["paired_new_minus_existing"][new_arm] = {}
        for metric in METRIC_NAMES:
            new = np.asarray([float(row[f"{new_arm}_{metric}"]) for row in rows], dtype=np.float64)
            summaries["paired_new_minus_existing"][new_arm][metric] = {}
            for base_arm in BASE_ARMS:
                base = np.asarray([float(row[f"{base_arm}_{metric}"]) for row in rows], dtype=np.float64)
                delta = new - base
                ci_seed += 1
                summaries["paired_new_minus_existing"][new_arm][metric][base_arm] = {
                    "mean_delta": float(delta.mean()),
                    "median_delta": float(np.median(delta)),
                    "bootstrap_95ci": bootstrap_ci(delta, ci_seed),
                    "new_arm_wins": int((delta > 0).sum()),
                    "existing_arm_wins": int((delta < 0).sum()),
                    "ties": int((delta == 0).sum()),
                }
    return summaries


def import_dns_evaluator():
    spec = importlib.util.spec_from_file_location("p7_dnsmos_evaluator", DNS_EVALUATOR)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load DNSMOS evaluator: {DNS_EVALUATOR}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def read_csv_map(path: Path) -> dict[str, dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    mapped = {row["filename"]: row for row in rows}
    if len(rows) != len(mapped):
        raise ValueError(f"Duplicate filenames in {path}")
    return mapped


def metric_from_baselines(name: str, path: Path) -> dict[str, dict[str, float]]:
    source = read_csv_map(path)
    key = {
        "spectral_and_si_sdr": "si_sdr",
        "pesq_stoi": "pesq_stoi",
        "dnsmos": "dnsmos",
    }[name]
    result: dict[str, dict[str, float]] = {}
    for filename, row in source.items():
        result[filename] = {}
        if key == "si_sdr":
            for arm in BASE_ARMS:
                result[filename][f"{arm}_si_sdr"] = float(row[f"{arm}_si_sdr"])
        elif key == "pesq_stoi":
            for arm in BASE_ARMS:
                result[filename][f"{arm}_pesq_wb"] = float(row[f"{arm}_pesq_wb"])
                result[filename][f"{arm}_stoi"] = float(row[f"{arm}_stoi"])
        else:
            for arm in BASE_ARMS:
                for metric in ("p808", "sig", "bak", "ovrl"):
                    result[filename][f"{arm}_dnsmos_{metric}"] = float(row[f"{arm}_{metric}"])
    return result


def summarize_reconstruction(rows: list[dict[str, Any]]) -> dict[str, Any]:
    keys = (
        "source_frontend_relative_rmse", "source_off_relative_rmse",
        "magP_phaseG_stft_projection_relative_l2", "magG_phaseP_stft_projection_relative_l2",
    )
    return {
        key: {
            "mean": float(np.mean([float(r[key]) for r in rows])),
            "median": float(np.median([float(r[key]) for r in rows])),
            "max": float(np.max([float(r[key]) for r in rows])),
        }
        for key in keys
    }


def write_per_utterance(rows: list[dict[str, Any]]) -> None:
    path = OUT / "per_utterance_metrics.csv"
    fields = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def report_text(metrics: dict[str, Any], cfg: dict[str, Any], status: str) -> str:
    lines = [
        "# P7-C fixed amplitude/phase cross diagnostic — 2026-10-17",
        "",
        f"**Status:** {status}",
        "",
        "## Scope",
        "",
        "This zero-training diagnostic reuses the existing 32-file P7-C development manifest. It constructs exactly two fixed arms from the paired Frontend-only waveform `Y_pre` and Router-off waveform `X_gen`: `|P|∠G` (`magP_phaseG`) and `|G|∠P` (`magG_phaseP`), where `P=STFT(Y_pre)` and `G=STFT(X_gen)`. No mixing coefficient, oracle selector, threshold, or model parameter was searched.",
        "",
        "The split has already been used for development screening and is not an independent blind test. Clean audio was used only as the reference for offline metric evaluation.",
        "",
        "## Construction and checks",
        "",
        f"- STFT: {SR} Hz, `n_fft={N_FFT}`, `hop={HOP}`, periodic Hann, centered; same Torch STFT/iSTFT definition as the prior P7-C analysis.",
        "- Both arms were saved as 32-bit floating point WAVs after iSTFT. Sample lengths and finite values were checked per file.",
        "- Original Frontend and Router-off iSTFT round-trip errors and each cross-arm STFT projection consistency residual are included in `metrics.json` and `per_utterance_metrics.csv`.",
        f"- DNSMOS request: `{cfg.get('dnsmos', {}).get('requested_device', 'not reached')}`; resolved runtime/provider: `{json.dumps(cfg.get('dnsmos', {}).get('runtime_info', {}), ensure_ascii=False)}`.",
        "- This run reuses the existing DNSMOS evaluator implementation and did not modify it.",
        "",
        "## Mean scores",
        "",
        "| Arm | PESQ-WB | STOI | SI-SDR (dB) | DNSMOS P.808 | SIG | BAK | OVRL |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    if status == "complete":
        summaries = metrics["summary"]["arms"]
        order = ("magP_phaseG", "magG_phaseP", "frontend", "off", "router")
        for arm in order:
            s = summaries[arm]
            lines.append(
                f"| {arm} | {s['pesq_wb']['mean']:.4f} | {s['stoi']['mean']:.4f} | {s['si_sdr']['mean']:.4f} | "
                f"{s['dnsmos_p808']['mean']:.4f} | {s['dnsmos_sig']['mean']:.4f} | "
                f"{s['dnsmos_bak']['mean']:.4f} | {s['dnsmos_ovrl']['mean']:.4f} |"
            )
        lines.extend(["", "The `metrics.json` contains paired per-utterance deltas and 10,000-replicate bootstrap 95% intervals for each new arm against each of Frontend-only, Router-off, and Router outputs.", ""])
        lines.append("## Waveform and STFT consistency"); lines.append("")
        for key, values in metrics["reconstruction_summary"].items():
            lines.append(f"- `{key}`: mean {values['mean']:.3e}, median {values['median']:.3e}, max {values['max']:.3e}.")
        lines.extend(["", "## Interpretation limits", "", "These fixed crossings test whether the observed amplitude and phase strengths combine into useful waveforms under a specific iSTFT. They do not establish a causal mechanism, deployable rule, or generalization beyond this previously screened 32-utterance split. Any score change must be interpreted across all metrics; clean and DNSMOS were not used to select or tune an arm. No training, dataset edits, or checkpoint edits were performed.", ""])
    else:
        lines.extend(["", "The run stopped before completion. See `metrics.json` for the first recorded error and completed-file count. No blind retry was performed.", ""])
    return "\n".join(lines)


def initial_config() -> dict[str, Any]:
    stat = os.statvfs(ROOT)
    return {
        "experiment_id": "storm_error_attribution_p7_20261017_amp_phase_cross_20261017",
        "status": "running",
        "purpose": "Fixed |P|∠G and |G|∠P waveform diagnostics; zero training",
        "source_manifest": str(MANIFEST.relative_to(ROOT)),
        "source_manifest_sha256": sha256(MANIFEST),
        "n_expected": 32,
        "split_warning": "Previously used development split; not an independent blind test.",
        "inputs": {
            "frontend_y_pre": str(FRONT.relative_to(ROOT)),
            "router_off_x_gen": str(OFF.relative_to(ROOT)),
            "clean_reference": "clean_path entries in source manifest; offline evaluation only",
            "router_baseline_metrics": "existing paired metrics from frontend_attribution; Router audio is not an input to crossing construction",
        },
        "arms": {
            "magP_phaseG": "abs(STFT(frontend-only Y_pre)) * exp(1j * angle(STFT(Router-off X_gen)))",
            "magG_phaseP": "abs(STFT(Router-off X_gen)) * exp(1j * angle(STFT(frontend-only Y_pre)))",
        },
        "stft_istft": {
            "sample_rate_hz": SR,
            "n_fft": N_FFT,
            "hop_length": HOP,
            "win_length": N_FFT,
            "window": "periodic Hann",
            "center": True,
            "torch_stft_pad_mode": "reflect (PyTorch default)",
            "istft_length": "exact input sample count from manifest audio",
        },
        "audio_output": {"format": "WAV", "subtype": "FLOAT", "files_per_arm": 32},
        "metrics": ["PESQ-WB", "STOI", "SI-SDR", "DNSMOS P.808/SIG/BAK/OVRL"],
        "dnsmos": {"requested_device": "auto", "runtime_info": {}, "feature_extraction": "TorchMetrics librosa mel preprocessing remains CPU-side"},
        "bootstrap": {"replicates": BOOTSTRAP_REPS, "seed": BOOTSTRAP_SEED, "scope": "exploratory paired mean-difference intervals"},
        "search_or_selection": "none; exactly the two pre-registered crossings",
        "training": False,
        "dataset_modified": False,
        "checkpoint_modified": False,
        "source_audio_modified": False,
        "available_disk_bytes_before_run": int(stat.f_bavail * stat.f_frsize),
        "software": {"python": sys.version.split()[0], "platform": platform.platform(), "torch": torch.__version__, "numpy": np.__version__, "soundfile": sf.__version__},
    }


def run() -> tuple[dict[str, Any], dict[str, Any]]:
    if OUT.exists():
        # The launcher may have placed this script in the otherwise-empty
        # preflight directory. Never accept an existing result artifact.
        leftovers = [p for p in OUT.iterdir() if p.name not in {Path(__file__).name, "run.log"}]
        if leftovers:
            raise FileExistsError(f"Refusing to overwrite existing output directory: {OUT}")
    else:
        OUT.mkdir(parents=True, exist_ok=False)
    cfg = initial_config()
    write_config(cfg)
    started = time.time()
    rows: list[dict[str, Any]] = []
    audio_records: list[dict[str, Any]] = []
    try:
        avail = int(os.statvfs(ROOT).f_bavail * os.statvfs(ROOT).f_frsize)
        if avail < 500_000_000:
            raise RuntimeError(f"Insufficient free disk space for outputs: {avail} bytes")
        torch.set_num_threads(1)

        with MANIFEST.open(newline="", encoding="utf-8") as stream:
            entries = list(csv.DictReader(stream))
        if len(entries) != 32:
            raise ValueError(f"Expected 32 manifest entries, got {len(entries)}")
        names = [entry["filename"] for entry in entries]
        if len(names) != len(set(names)):
            raise ValueError("Duplicate filenames in source manifest")
        baseline_maps = {name: read_csv_map(path) for name, path in BASE_METRIC_CSVS.items()}
        for source, mapped in baseline_maps.items():
            if set(mapped) != set(names):
                raise ValueError(f"Existing {source} metric filenames do not match source manifest")
        baseline_rows: dict[str, dict[str, float]] = {filename: {} for filename in names}
        for source in baseline_maps:
            converted = metric_from_baselines(source, BASE_METRIC_CSVS[source])
            for filename, values in converted.items():
                baseline_rows[filename].update(values)
        for filename, item in baseline_rows.items():
            if not filename:
                raise ValueError("Empty filename in existing metrics")

        # Validate the P7-C source audio hashes against the prior paired-audio ledger.
        previous_audio = read_csv_map(BASE_DIR / "per_utterance_metrics.csv")
        dns_meta = json.loads((BASE_DIR / "dnsmos_metrics.json").read_text(encoding="utf-8"))
        pesq_meta = json.loads((BASE_DIR / "pesq_stoi_metrics.json").read_text(encoding="utf-8"))
        expected_manifest_hash = sha256(MANIFEST)
        for meta_name, meta in (("DNSMOS", dns_meta), ("PESQ/STOI", pesq_meta)):
            if meta.get("manifest_sha256") != expected_manifest_hash:
                raise ValueError(f"Existing {meta_name} metrics use a different manifest")

        dns_eval = import_dns_evaluator()
        resolved = dns_eval.resolve_device("auto")
        cfg["dnsmos"]["resolved_device"] = str(resolved)
        cfg["dnsmos"]["runtime_info"] = dict(dns_eval._RUNTIME_INFO)
        if resolved.type != "cuda":
            raise RuntimeError(f"DNSMOS auto did not resolve to CUDA: {dns_eval._RUNTIME_INFO}")
        if any("CUDAExecutionProvider" not in providers for providers in dns_eval._RUNTIME_INFO["onnx_session_providers"].values()):
            raise RuntimeError(f"One or more DNSMOS sessions are not configured with CUDA: {dns_eval._RUNTIME_INFO}")
        write_config(cfg)

        for index, entry in enumerate(entries, 1):
            name = entry["filename"]
            paths = {
                "clean": Path(entry["clean_path"]),
                "frontend": FRONT / name,
                "off": OFF / name,
                "router": ROUTER / name,
            }
            for key, path in paths.items():
                if not path.is_file():
                    raise FileNotFoundError(path)
            hashes = {key: sha256(path) for key, path in paths.items()}
            if hashes["clean"] != entry["clean_sha256"]:
                raise ValueError(f"Clean hash mismatch against manifest for {name}")
            if hashes["off"] != entry["router_off_sha256"]:
                raise ValueError(f"Router-off hash mismatch against manifest for {name}")
            # Existing spectral rows store the same per-file scale/length audit.
            prior = previous_audio[name]
            if int(prior["length"]) != int(entry["noisy_samples"]):
                raise ValueError(f"Existing P7-C length audit disagrees for {name}")

            waves: dict[str, np.ndarray] = {}
            sample_rates: dict[str, int] = {}
            for key, path in paths.items():
                waves[key], sample_rates[key] = load_wave(path)
            lengths = {key: int(wave.size) for key, wave in waves.items()}
            if len(set(lengths.values())) != 1:
                raise ValueError(f"Unaligned input lengths for {name}: {lengths}")
            expected_len = int(entry["noisy_samples"])
            if lengths["clean"] != expected_len:
                raise ValueError(f"Length mismatch to manifest for {name}: {lengths['clean']} != {expected_len}")
            if hashes["frontend"] == hashes["off"]:
                raise ValueError(f"Frontend and Router-off source unexpectedly identical: {name}")

            # P=frontend-only, G=Router-off; exact fixed magnitude/phase crossings.
            P = stft(waves["frontend"])
            G = stft(waves["off"])
            if P.shape != G.shape:
                raise ValueError(f"STFT shape mismatch for {name}: {P.shape} vs {G.shape}")
            z_magP_phaseG = torch.polar(torch.abs(P), torch.angle(G))
            z_magG_phaseP = torch.polar(torch.abs(G), torch.angle(P))
            if not bool(torch.isfinite(z_magP_phaseG.real).all() and torch.isfinite(z_magP_phaseG.imag).all()):
                raise ValueError(f"Nonfinite |P| angle(G) spectrum for {name}")
            if not bool(torch.isfinite(z_magG_phaseP.real).all() and torch.isfinite(z_magG_phaseP.imag).all()):
                raise ValueError(f"Nonfinite |G| angle(P) spectrum for {name}")

            rec_P = waveform_reconstruction(waves["frontend"], istft(P, expected_len))
            rec_G = waveform_reconstruction(waves["off"], istft(G, expected_len))
            row: dict[str, Any] = {
                "filename": name,
                "uttid": entry["uttid"],
                "length_samples": expected_len,
                "sample_rate_hz": SR,
                "source_clean_sha256": hashes["clean"],
                "source_frontend_sha256": hashes["frontend"],
                "source_off_sha256": hashes["off"],
                "source_router_sha256": hashes["router"],
                "source_frontend_peak": float(np.max(np.abs(waves["frontend"]))),
                "source_off_peak": float(np.max(np.abs(waves["off"]))),
                "source_frontend_relative_rmse": rec_P["relative_rmse"],
                "source_frontend_max_abs_error": rec_P["max_abs_error"],
                "source_off_relative_rmse": rec_G["relative_rmse"],
                "source_off_max_abs_error": rec_G["max_abs_error"],
            }
            source_baseline = baseline_rows[name]
            row.update(source_baseline)

            for arm, spec in (("magP_phaseG", z_magP_phaseG), ("magG_phaseP", z_magG_phaseP)):
                waveform = istft(spec, expected_len)
                projection = spectral_consistency(spec, waveform)
                target_path = ARM_DIRS[arm] / name
                target_path.parent.mkdir(parents=True, exist_ok=True)
                sf.write(target_path, waveform, SR, subtype="FLOAT", format="WAV")
                saved, saved_sr = load_wave(target_path)
                if saved_sr != SR or saved.size != expected_len:
                    raise ValueError(f"Saved output length/sample-rate mismatch: {target_path}")
                if not np.isfinite(saved).all():
                    raise ValueError(f"Saved output contains nonfinite values: {target_path}")
                row[f"{arm}_peak"] = float(np.max(np.abs(saved)))
                row[f"{arm}_rms"] = float(np.sqrt(np.mean(saved.astype(np.float64) ** 2)))
                row[f"{arm}_stft_projection_relative_l2"] = projection["stft_projection_relative_l2"]
                row[f"{arm}_stft_projection_max_abs_error"] = projection["stft_projection_max_abs_error"]
                row[f"{arm}_si_sdr"] = si_sdr(waves["clean"], saved)
                row[f"{arm}_pesq_wb"] = float(pesq(SR, waves["clean"], saved, "wb"))
                row[f"{arm}_stoi"] = float(stoi(waves["clean"], saved, SR, extended=False))
                dns_input = dns_eval.load(target_path)
                dns_vals = dns_eval.mos(dns_input, device="auto")
                runtime_info = dict(dns_eval._RUNTIME_INFO)
                if runtime_info.get("device") != "cuda:0" or runtime_info.get("execution_provider") != "CUDAExecutionProvider":
                    raise RuntimeError(f"DNSMOS did not remain on CUDA for {name} / {arm}: {runtime_info}")
                for metric, value in zip(("p808", "sig", "bak", "ovrl"), dns_vals):
                    row[f"{arm}_dnsmos_{metric}"] = float(value)
                row[f"{arm}_dns_resolved_device"] = str(runtime_info["device"])
                row[f"{arm}_dns_execution_provider"] = str(runtime_info["execution_provider"])
                audio_records.append({
                    "filename": name,
                    "uttid": entry["uttid"],
                    "arm": arm,
                    "path": str(target_path.relative_to(ROOT)),
                    "sha256": sha256(target_path),
                    "sample_rate_hz": saved_sr,
                    "length_samples": int(saved.size),
                    "subtype": "FLOAT",
                    **projection,
                })

            rows.append(row)
            if index % 4 == 0 or index == len(entries):
                print(f"evaluated {index}/{len(entries)} utterances", flush=True)

        summary = metric_summary(rows)
        reconstruction_summary = summarize_reconstruction(rows)
        dns_final_info = dict(dns_eval._RUNTIME_INFO)
        if dns_final_info.get("device") != "cuda:0":
            raise RuntimeError(f"DNSMOS resolved away from CUDA during run: {dns_final_info}")
        cfg.update({
            "status": "complete",
            "runtime_seconds": time.time() - started,
            "dnsmos": {
                **cfg["dnsmos"],
                "requested_device": "auto",
                "resolved_device": dns_final_info["device"],
                "runtime_info": dns_final_info,
                "inference_arms": list(ARM_DIRS),
                "cpu_fallback_observed": False,
            },
            "reconstruction_summary": reconstruction_summary,
        })
        metrics = {
            "experiment_id": cfg["experiment_id"],
            "status": "complete",
            "n": len(rows),
            "manifest_sha256": cfg["source_manifest_sha256"],
            "stft_istft": cfg["stft_istft"],
            "audio_arms": {key: str(path.relative_to(ROOT)) for key, path in ARM_DIRS.items()},
            "dnsmos_runtime": cfg["dnsmos"],
            "reconstruction_summary": reconstruction_summary,
            "summary": summary,
            "bootstrap": cfg["bootstrap"],
            "runtime_seconds": cfg["runtime_seconds"],
            "training": False,
            "source_audio_modified": False,
            "dataset_modified": False,
            "checkpoint_modified": False,
        }
        write_per_utterance(rows)
        write_json(OUT / "paired_audio_manifest.json", audio_records)
        write_json(OUT / "metrics.json", metrics)
        write_config(cfg)
        (OUT / "experiment_report.md").write_text(report_text(metrics, cfg, "complete"), encoding="utf-8")
        return cfg, metrics
    except Exception as exc:
        cfg.update({"status": "failed", "runtime_seconds": time.time() - started})
        write_config(cfg)
        failure = {
            "experiment_id": cfg["experiment_id"],
            "status": "failed",
            "completed_utterances": len(rows),
            "error_type": type(exc).__name__,
            "error": str(exc),
            "traceback": traceback.format_exc(),
            "runtime_seconds": time.time() - started,
            "source_manifest_sha256": cfg.get("source_manifest_sha256"),
            "training": False,
            "source_audio_modified": False,
            "dataset_modified": False,
            "checkpoint_modified": False,
        }
        write_json(OUT / "metrics.json", failure)
        (OUT / "experiment_report.md").write_text(report_text(failure, cfg, "failed"), encoding="utf-8")
        raise


if __name__ == "__main__":
    run()
