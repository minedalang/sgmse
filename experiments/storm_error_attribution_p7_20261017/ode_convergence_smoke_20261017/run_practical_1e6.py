#!/usr/bin/env python3
"""Bounded single-utterance RK45 1e-6 follow-up for P7-A.

This reuses the existing checkpoint, manifest row, saved x_T and 1e-5 FLOAT
WAV. It writes only a new 1e-6 WAV and an independent supplemental metrics JSON.
"""
from __future__ import annotations

import csv
import json
import math
import platform
import sys
import threading
import time
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf
import torch


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.dont_write_bytecode = True
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from run_ode_convergence_smoke import (  # noqa: E402
    CHECKPOINT,
    EXPECTED_CHECKPOINT_SHA256,
    EXPECTED_FILES,
    EXPECTED_MANIFEST_SHA256,
    MANIFEST,
    check_finite_tensor,
    load_model,
    prepare_pair,
    sha256_file,
)


OUT_WAV = HERE / "rk45_rtol_atol_1e-06" / "000_fileid_2673.wav"
OUT_METRICS = HERE / "supplemental_metrics_1e-6.json"
BASE_METRICS = HERE / "metrics.json"
BASE_WAV = HERE / "rk45_rtol_atol_1e-05" / "000_fileid_2673.wav"
XT_PATH = HERE / "shared_xT" / "000_fileid_2673.pt"
TOLERANCE = 1e-6
SAMPLE_RATE = 16000


def write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def relative_l2(reference: np.ndarray, estimate: np.ndarray) -> float:
    ref = np.asarray(reference, dtype=np.complex128 if np.iscomplexobj(reference) else np.float64)
    est = np.asarray(estimate, dtype=ref.dtype)
    denom = max(float(np.linalg.norm(ref.ravel())), 1e-30)
    return float(np.linalg.norm((est - ref).ravel()) / denom)


def first_manifest_row() -> dict[str, str]:
    if sha256_file(MANIFEST) != EXPECTED_MANIFEST_SHA256:
        raise ValueError("P7 manifest SHA256 mismatch")
    with MANIFEST.open(newline="", encoding="utf-8") as stream:
        row = next(csv.DictReader(stream))
    if row.get("filename") != EXPECTED_FILES[0] or row.get("filename") != "000_fileid_2673.wav":
        raise ValueError(f"Unexpected first P7 manifest row: {row.get('filename')}")
    input_path = Path(row["noisy_path"])
    if not input_path.is_file() or sha256_file(input_path) != row["noisy_sha256"]:
        raise ValueError(f"P7 noisy input is missing or hash-mismatched: {input_path}")
    return row


def main() -> None:
    if sha256_file(CHECKPOINT) != EXPECTED_CHECKPOINT_SHA256:
        raise ValueError("Fixed 80k checkpoint SHA256 mismatch")
    if not XT_PATH.is_file() or not BASE_WAV.is_file() or not BASE_METRICS.is_file():
        raise FileNotFoundError("Required existing x_T, 1e-5 WAV, or metrics.json is missing")
    if OUT_WAV.exists() or OUT_METRICS.exists():
        raise FileExistsError("Refusing to overwrite a prior 1e-6 supplemental result")

    base_metrics_hash_before = sha256_file(BASE_METRICS)
    xT_hash = sha256_file(XT_PATH)
    base_wav_hash = sha256_file(BASE_WAV)
    prior_metrics = json.loads(BASE_METRICS.read_text(encoding="utf-8"))
    prior_utt = next(x for x in prior_metrics["utterances"] if x.get("filename") == "000_fileid_2673.wav")
    prior_run = next(x for x in prior_utt["completed_runs"] if x.get("tolerance") == 1e-5)
    if prior_run.get("wav_sha256") != base_wav_hash:
        raise ValueError("Existing 1e-5 WAV hash does not match metrics.json")
    if prior_utt.get("shared_xT", {}).get("sha256") != xT_hash:
        raise ValueError("Existing shared x_T hash does not match metrics.json")

    row = first_manifest_row()
    device = torch.device("cuda")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the bounded P7-A follow-up")

    run_started = time.perf_counter()
    model, loaded_parameters = load_model(device)
    model_load_seconds = time.perf_counter() - run_started
    prepare_started = time.perf_counter()
    noisy_waveform, y_main, y_cond, norm_factor = prepare_pair(model, row, device)
    prepare_seconds = time.perf_counter() - prepare_started

    x_T = torch.load(XT_PATH, map_location=device, weights_only=True)
    if not isinstance(x_T, torch.Tensor):
        raise TypeError(f"Saved shared x_T is not a tensor: {type(x_T).__name__}")
    x_T = x_T.to(device=device).detach().contiguous()
    check_finite_tensor("reused shared x_T", x_T)
    if tuple(x_T.shape) != tuple(y_main.shape):
        raise ValueError(f"Saved x_T shape {tuple(x_T.shape)} does not match Y_pre {tuple(y_main.shape)}")

    base_wave, base_sr = sf.read(BASE_WAV, dtype="float32", always_2d=False)
    if base_sr != SAMPLE_RATE or base_wave.ndim != 1 or not np.isfinite(base_wave).all():
        raise ValueError("Existing 1e-5 reference WAV is not finite mono 16 kHz audio")
    if base_wave.shape != noisy_waveform.shape:
        raise ValueError("Existing 1e-5 reference length does not match the input utterance")

    ode_sde = model.sde.copy()
    ode_sde.N = int(model.sde.N)

    def fixed_prior(shape, _y):
        if tuple(shape) != tuple(x_T.shape):
            raise ValueError(f"Saved x_T shape {tuple(x_T.shape)} != requested {tuple(shape)}")
        return x_T.clone()

    ode_sde.prior_sampling = fixed_prior
    score_calls = [0]

    def count_score_forward(_module, _inputs, _output):
        score_calls[0] += 1

    progress: list[dict[str, float | int]] = []
    stop_heartbeat = threading.Event()
    solve_started_at = [0.0]

    def heartbeat() -> None:
        while not stop_heartbeat.wait(30.0):
            elapsed = time.perf_counter() - solve_started_at[0]
            entry: dict[str, float | int] = {
                "elapsed_seconds": round(elapsed, 3),
                "score_forward_calls": int(score_calls[0]),
            }
            progress.append(entry)
            print(
                f"[P7-A 1e-6 progress] elapsed={elapsed:.0f}s "
                f"score_forward_calls={score_calls[0]}",
                flush=True,
            )

    hook = model.dnn.register_forward_hook(count_score_forward)
    stop_reason = None
    nfev: int | None = None
    sample: torch.Tensor | None = None
    solve_elapsed = 0.0
    watcher = threading.Thread(target=heartbeat, name="ode-progress", daemon=True)
    watcher.start()
    try:
        from sgmse.sampling import get_ode_sampler

        score_fn = lambda x, y, t: model(x, y, t, y_cond=y_cond)
        sampler = get_ode_sampler(
            ode_sde,
            score_fn,
            y=y_main,
            denoise=False,
            rtol=TOLERANCE,
            atol=TOLERANCE,
            method="RK45",
            eps=float(model.t_eps),
            device=str(device),
        )
        print("[P7-A 1e-6] starting RK45 solve for 000_fileid_2673.wav", flush=True)
        solve_started_at[0] = time.perf_counter()
        solve_start = solve_started_at[0]
        sample, nfev_value = sampler()
        solve_elapsed = time.perf_counter() - solve_start
        nfev = int(nfev_value)
    except KeyboardInterrupt:
        solve_elapsed = time.perf_counter() - solve_started_at[0] if solve_started_at[0] else 0.0
        stop_reason = "SIGINT before endpoint"
        print("[P7-A 1e-6] interrupted before endpoint", flush=True)
    finally:
        stop_heartbeat.set()
        watcher.join(timeout=2.0)
        hook.remove()

    common = {
        "experiment_id": "storm_error_attribution_p7_20261017_ode_convergence_smoke_20261017",
        "supplement": "single-utterance practical 1e-6 RK45 probability-flow ODE cost and delta check",
        "status": "interrupted_before_endpoint" if stop_reason else "completed",
        "checkpoint": str(CHECKPOINT),
        "checkpoint_sha256": EXPECTED_CHECKPOINT_SHA256,
        "checkpoint_global_step": 80000,
        "manifest": str(MANIFEST),
        "manifest_sha256": EXPECTED_MANIFEST_SHA256,
        "utterance": {
            "filename": row["filename"],
            "uttid": row["uttid"],
            "noisy_path": row["noisy_path"],
            "noisy_sha256": row["noisy_sha256"],
            "sample_rate_hz": SAMPLE_RATE,
            "input_samples": int(noisy_waveform.size),
        },
        "initial_state": {
            "path": str(XT_PATH.relative_to(ROOT)),
            "sha256": xT_hash,
            "reused_without_rewrite": True,
        },
        "sampler": {
            "implementation": "sgmse.sampling.get_ode_sampler",
            "process": "probability_flow",
            "solver": "SciPy RK45",
            "rtol": TOLERANCE,
            "atol": TOLERANCE,
            "denoise": False,
            "integration_interval": [1.0, float(model.t_eps)],
            "initial_state_policy": "reuse the existing saved 000_fileid_2673 x_T tensor",
        },
        "runtime": {
            "python": platform.python_version(),
            "torch": str(torch.__version__),
            "numpy": np.__version__,
            "scipy": __import__("scipy").__version__,
            "device": str(device),
            "model_load_seconds": model_load_seconds,
            "input_preparation_seconds": prepare_seconds,
            "solve_elapsed_seconds": solve_elapsed,
            "total_elapsed_seconds": time.perf_counter() - run_started,
            "progress_samples": progress,
        },
        "reference_1e-5": {
            "wav_path": str(BASE_WAV.relative_to(ROOT)),
            "wav_sha256": base_wav_hash,
            "metrics_json_sha256_before": base_metrics_hash_before,
        },
        "loaded_model_parameters": {name: actual for name, (actual, _expected) in loaded_parameters.items()},
        "training": False,
        "quality_evaluation": False,
        "dns_mos_or_other_speech_quality_metrics": False,
        "checkpoint_or_source_data_modified": False,
    }
    if stop_reason:
        common["interruption"] = {
            "reason": stop_reason,
            "nfev": None,
            "score_dnn_forward_calls_observed_before_interrupt": int(score_calls[0]),
            "endpoint_returned": False,
            "wav_written": False,
        }
        write_json(OUT_METRICS, common)
        raise SystemExit(130)

    if sample is None or nfev is None:
        raise RuntimeError("ODE sampler returned without an endpoint or NFE")
    check_finite_tensor("1e-6 endpoint spectrum", sample)
    if nfev != score_calls[0]:
        common["nfev_count_validation"] = {
            "nfev": nfev,
            "score_dnn_forward_calls": score_calls[0],
            "matches": False,
        }
        common["status"] = "failed_nfev_hook_count_mismatch"
        write_json(OUT_METRICS, common)
        raise RuntimeError(f"SciPy nfev={nfev} does not match hooked DNN calls={score_calls[0]}")

    with torch.inference_mode():
        output = model.to_audio(sample.squeeze(), length=int(noisy_waveform.size)).detach() * norm_factor
    check_finite_tensor("1e-6 output waveform", output)
    generated_wave = output.squeeze().cpu().numpy().astype(np.float32, copy=False)
    if generated_wave.shape != noisy_waveform.shape or not np.isfinite(generated_wave).all():
        raise ValueError("1e-6 output waveform has an invalid shape or nonfinite values")
    with torch.inference_mode():
        generated_stft = model._stft(output.reshape(1, -1)).detach()
        reference_wave_tensor = torch.from_numpy(np.ascontiguousarray(base_wave)).to(device=device).reshape(1, -1)
        reference_stft = model._stft(reference_wave_tensor).detach()
    check_finite_tensor("1e-6 reconstructed complex STFT", generated_stft)
    check_finite_tensor("1e-5 reference complex STFT", reference_stft)

    clipped = np.clip(generated_wave, -1.0, 1.0)
    OUT_WAV.parent.mkdir(parents=True, exist_ok=False)
    sf.write(OUT_WAV, clipped, SAMPLE_RATE, subtype="FLOAT")
    wave_delta = relative_l2(base_wave, generated_wave)
    stft_delta = relative_l2(reference_stft.cpu().numpy(), generated_stft.cpu().numpy())
    common.update({
        "nfev": nfev,
        "score_dnn_forward_calls": int(score_calls[0]),
        "nfev_matches_hook_count": True,
        "endpoint_spectrum_finite": True,
        "endpoint_waveform_finite": True,
        "output": {
            "wav_path": str(OUT_WAV.relative_to(ROOT)),
            "wav_sha256": sha256_file(OUT_WAV),
            "wav_subtype": "FLOAT",
            "samples": int(generated_wave.size),
            "unclipped_absmax": float(np.max(np.abs(generated_wave))),
            "clipped_sample_count": int(np.count_nonzero(clipped != generated_wave)),
        },
        "comparison_to_1e-5": {
            "reference_wav_sha256": base_wav_hash,
            "waveform_relative_l2": wave_delta,
            "complex_stft_relative_l2": stft_delta,
            "complex_stft_parameters": {
                "implementation": "model._stft",
                "n_fft": 510,
                "hop_length": 128,
                "window": "periodic Hann",
                "center": True,
            },
            "reference_is_1e-5_saved_FLOAT_wav": True,
        },
        "immutability_checks": {
            "shared_xT_sha256_after": sha256_file(XT_PATH),
            "metrics_json_sha256_after": sha256_file(BASE_METRICS),
            "1e-5_wav_sha256_after": sha256_file(BASE_WAV),
            "shared_xT_unchanged": sha256_file(XT_PATH) == xT_hash,
            "metrics_json_unchanged": sha256_file(BASE_METRICS) == base_metrics_hash_before,
            "1e-5_wav_unchanged": sha256_file(BASE_WAV) == base_wav_hash,
        },
    })
    if not all(common["immutability_checks"][key] for key in ("shared_xT_unchanged", "metrics_json_unchanged", "1e-5_wav_unchanged")):
        raise RuntimeError("A protected pre-existing artifact changed unexpectedly")
    write_json(OUT_METRICS, common)
    print(
        f"[P7-A 1e-6] completed nfev={nfev} score_calls={score_calls[0]} "
        f"solve={solve_elapsed:.2f}s waveform_rel_l2={wave_delta:.8g} "
        f"complex_stft_rel_l2={stft_delta:.8g}",
        flush=True,
    )


if __name__ == "__main__":
    main()
