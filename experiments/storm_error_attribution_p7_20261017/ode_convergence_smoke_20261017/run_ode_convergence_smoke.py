#!/usr/bin/env python3
"""Four-utterance shared-prior ODE tolerance smoke for the fixed P7 checkpoint.

This uses the repository probability-flow ODE helper and supplies each run with
the same saved OUVE prior draw. It writes WAVs and integration metadata only;
it does not train or compute speech quality metrics.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf
import torch


ROOT = Path(__file__).resolve().parents[3]
P7 = ROOT / "experiments/storm_error_attribution_p7_20261017"
OUT = Path(__file__).resolve().parent
CHECKPOINT = ROOT / "joint_train_logs/sb61i0gj_RRSD/epoch=118-step=80000.ckpt"
MANIFEST = P7 / "baseline_manifest.csv"
EXPECTED_CHECKPOINT_SHA256 = "ab50510ca7fd80798b70f3f0b5365b17448a5f98c75d38755b82212ae2a119c9"
EXPECTED_MANIFEST_SHA256 = "77b64db9da6a1ff372a22dcda4a6316743133f383103aaffb3dbce2c671c178e"
EXPECTED_FILES = (
    "000_fileid_2673.wav",
    "001_fileid_2972.wav",
    "002_fileid_7247.wav",
    "003_fileid_4431.wav",
)
TOLERANCES = (1e-3, 1e-5, 1e-7)
SAMPLE_RATE = 16000
SEED_BASE = 20261017


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def check_finite_tensor(name: str, value: torch.Tensor) -> None:
    if torch.is_complex(value):
        finite = torch.isfinite(value.real).all() and torch.isfinite(value.imag).all()
    else:
        finite = torch.isfinite(value).all()
    if not bool(finite):
        raise FloatingPointError(f"{name} contains nonfinite values")


def read_manifest_rows() -> list[dict[str, str]]:
    with MANIFEST.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    if len(rows) != 32:
        raise ValueError(f"Expected the P7 32-row manifest, found {len(rows)} rows")
    chosen = rows[: len(EXPECTED_FILES)]
    if tuple(row["filename"] for row in chosen) != EXPECTED_FILES:
        raise ValueError("The first four P7 manifest rows no longer match the fixed smoke set")
    for row in chosen:
        path = Path(row["noisy_path"])
        if not path.is_file():
            raise FileNotFoundError(path)
        if sha256_file(path) != row["noisy_sha256"]:
            raise ValueError(f"Noisy input hash changed: {path}")
    return chosen


def load_model(device: torch.device):
    # This is the same strict checkpoint/DataModule load used by P7's
    # frontend-only export. No frontend override is supplied: checkpoint values
    # select and populate the StoRM frontend.
    from sgmse.joint_data_module import JointWaveDataModule
    from sgmse.joint_model import JointSemambaSgmseModel

    model = JointSemambaSgmseModel.load_from_checkpoint(
        str(CHECKPOINT),
        map_location="cpu",
        strict=True,
        data_module_cls=JointWaveDataModule,
        base_dir=".",
        batch_size=1,
        num_workers=0,
        dummy=True,
        weights_only=False,
    )
    model.eval(no_ema=False).to(device)

    dm = model.data_module
    checks = {
        "frontend": (model.frontend_name, "storm"),
        "backbone": (model.backbone, "ncsnpp_v2"),
        "sde": (model.sde.__class__.__name__, "OUVESDE"),
        "saved_sampler_type": (model.sde.sampler_type, "pc"),
        "train_type": (model.train_type, "mode2"),
        "mode2_include_noisy_main": (bool(model.mode2_include_noisy_main), True),
        "sample_rate": (int(model.sr), SAMPLE_RATE),
        "N": (int(model.sde.N), 30),
        "theta": (float(model.sde.theta), 1.5),
        "sigma_min": (float(model.sde.sigma_min), 0.05),
        "sigma_max": (float(model.sde.sigma_max), 0.5),
        "t_eps": (float(model.t_eps), 0.03),
        "loss_type": (model.loss_type, "score_matching"),
        "loss_weighting": (model.loss_weighting, "sigma^2"),
        "network_scaling": (model.network_scaling, "1/t"),
        "c_in": (model.c_in, "1"),
        "c_out": (model.c_out, "1"),
        "c_skip": (model.c_skip, "0"),
        "sigma_data": (float(model.sigma_data), 0.1),
        "n_fft": (int(dm.n_fft), 510),
        "hop_length": (int(dm.hop_length), 128),
        "num_frames": (int(dm.num_frames), 256),
        "spec_factor": (float(dm.spec_factor), 0.15),
        "spec_abs_exponent": (float(dm.spec_abs_exponent), 0.5),
        "normalize": (dm.normalize, "noisy"),
        "transform_type": (dm.transform_type, "exponent"),
    }
    mismatches = {
        name: {"loaded": actual, "expected": expected}
        for name, (actual, expected) in checks.items()
        if actual != expected
    }
    if mismatches:
        raise ValueError(f"Checkpoint/model configuration mismatch: {mismatches}")

    # The P7 checkpoint records a Hann analysis window; verify the loaded
    # periodic window instead of inferring it from the saved hparam string.
    expected_window = torch.hann_window(510, periodic=True, dtype=dm.window.dtype)
    if not torch.equal(dm.window.detach().cpu(), expected_window):
        raise ValueError("Loaded DataModule window is not the expected periodic Hann window")
    if bool(getattr(model.sde, "residual_anchor_enabled", False)):
        raise ValueError("P7-0 checkpoint unexpectedly enables residual anchoring")
    return model, checks


def match_length(value: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
    target = int(reference.size(-1))
    current = int(value.size(-1))
    if current == target:
        return value
    if current > target:
        start = (current - target) // 2
        return value[..., start : start + target]
    missing = target - current
    left = missing // 2
    return torch.nn.functional.pad(value, (left, missing - left))


def prepare_pair(model, row: dict[str, str], device: torch.device):
    from sgmse.util.other import pad_spec

    path = Path(row["noisy_path"])
    waveform, sample_rate = sf.read(path, dtype="float32", always_2d=False)
    if sample_rate != SAMPLE_RATE:
        raise ValueError(f"Expected existing 16 kHz P7 input, got {sample_rate}: {path}")
    if waveform.ndim == 2:
        waveform = waveform.mean(axis=-1, dtype=np.float32)
    if waveform.ndim != 1 or not np.isfinite(waveform).all():
        raise ValueError(f"Expected finite mono waveform: {path}")

    noisy = torch.from_numpy(np.ascontiguousarray(waveform)).to(device)[None, None, :]
    norm = noisy.abs().amax(dim=-1, keepdim=True).clamp_min(1e-12)
    with torch.inference_mode():
        pre = model._frontend_forward(noisy / norm) * norm
        pre = match_length(pre, noisy)
        if pre.shape != noisy.shape:
            raise ValueError(f"Frontend length alignment failed for {path}: {pre.shape} vs {noisy.shape}")

        # Match ScoreModel.enhance's mode2 path: Y_pre is the main diffusion
        # input; original noisy audio is the additional condition.
        pre_spec = torch.unsqueeze(model._forward_transform(model._stft(pre.squeeze(1) / norm.squeeze(1))), 0)
        noisy_spec = torch.unsqueeze(model._forward_transform(model._stft(noisy.squeeze(1) / norm.squeeze(1))), 0)
        y_main = pad_spec(pre_spec)
        y_cond = pad_spec(noisy_spec)
    check_finite_tensor("frontend waveform", pre)
    check_finite_tensor("Y_pre spectrum", y_main)
    check_finite_tensor("Y_noisy spectrum", y_cond)
    return waveform, y_main, y_cond, float(norm.item())


def save_shared_prior(model, y_main: torch.Tensor, seed: int, path: Path, device: torch.device) -> tuple[torch.Tensor, str]:
    devices = [device.index if device.index is not None else torch.cuda.current_device()] if device.type == "cuda" else []
    with torch.random.fork_rng(devices=devices):
        torch.manual_seed(seed)
        if device.type == "cuda":
            torch.cuda.manual_seed(seed)
        x_t = model.sde.prior_sampling(y_main.shape, y_main).to(device).detach().contiguous()
    check_finite_tensor("shared x_T", x_t)
    torch.save(x_t.cpu(), path)
    return x_t, sha256_file(path)


def output_delta(reference: torch.Tensor, estimate: torch.Tensor) -> dict[str, float]:
    diff = (estimate.detach().to(torch.complex128) - reference.detach().to(torch.complex128)).flatten()
    ref = reference.detach().to(torch.complex128).flatten()
    return {
        "max_abs": float(diff.abs().max().item()),
        "relative_l2": float((torch.linalg.vector_norm(diff) / torch.linalg.vector_norm(ref).clamp_min(1e-30)).item()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda", help="Torch inference device; default: cuda")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite only this experiment directory's known outputs")
    args = parser.parse_args()
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")

    if sha256_file(CHECKPOINT) != EXPECTED_CHECKPOINT_SHA256:
        raise ValueError(f"Checkpoint hash mismatch: {CHECKPOINT}")
    manifest_hash = sha256_file(MANIFEST)
    if manifest_hash != EXPECTED_MANIFEST_SHA256:
        raise ValueError(f"P7 manifest hash mismatch: {manifest_hash}")
    rows = read_manifest_rows()

    output_paths = [OUT / "metrics.json"]
    output_paths.extend(OUT / "shared_xT" / f"{Path(row['filename']).stem}.pt" for row in rows)
    for tolerance in TOLERANCES:
        directory = OUT / f"rk45_rtol_atol_{tolerance:.0e}"
        output_paths.extend(directory / row["filename"] for row in rows)
    existing = [path for path in output_paths if path.exists()]
    if existing and not args.overwrite:
        raise FileExistsError(
            "Outputs already exist; pass --overwrite to replace only the listed outputs: "
            + ", ".join(str(path) for path in existing)
        )

    start = time.perf_counter()
    model, loaded_parameters = load_model(device)
    model_config_seconds = time.perf_counter() - start
    manifest_info = []
    xT_dir = OUT / "shared_xT"
    xT_dir.mkdir(parents=True, exist_ok=True)
    tolerance_dirs = {
        tolerance: OUT / f"rk45_rtol_atol_{tolerance:.0e}" for tolerance in TOLERANCES
    }
    for directory in tolerance_dirs.values():
        directory.mkdir(parents=True, exist_ok=True)

    for index, row in enumerate(rows):
        print(f"[ODE smoke] preparing {index + 1}/{len(rows)} {row['filename']}", flush=True)
        waveform, y_main, y_cond, norm_factor = prepare_pair(model, row, device)
        xT_path = xT_dir / f"{Path(row['filename']).stem}.pt"
        x_T, xT_hash = save_shared_prior(model, y_main, SEED_BASE + index, xT_path, device)
        score_fn = lambda x, y, t, cond=y_cond: model(x, y, t, y_cond=cond)
        endpoint_specs: dict[float, torch.Tensor] = {}
        endpoint_waves: dict[float, np.ndarray] = {}
        endpoint_stfts: dict[float, torch.Tensor] = {}
        rows_for_utt = []

        for tolerance in TOLERANCES:
            print(
                f"[ODE smoke] start {row['filename']} rtol=atol={tolerance:.0e}",
                flush=True,
            )
            ode_sde = model.sde.copy()
            ode_sde.N = int(model.sde.N)

            # The stock helper always samples a new prior and ignores its z
            # argument. Bind this copied SDE's prior method to this utterance's
            # stored x_T so all tolerance runs start at exactly the same state.
            def fixed_prior(shape, _y, fixed=x_T):
                if tuple(shape) != tuple(fixed.shape):
                    raise ValueError(f"Stored x_T shape {tuple(fixed.shape)} != requested {tuple(shape)}")
                return fixed.clone()

            ode_sde.prior_sampling = fixed_prior
            call_count = [0]

            def count_score_forward(_module, _inputs, _output):
                call_count[0] += 1

            hook = model.dnn.register_forward_hook(count_score_forward)
            run_start = time.perf_counter()
            try:
                from sgmse.sampling import get_ode_sampler

                sampler = get_ode_sampler(
                    ode_sde,
                    score_fn,
                    y=y_main,
                    denoise=False,
                    rtol=tolerance,
                    atol=tolerance,
                    method="RK45",
                    eps=float(model.t_eps),
                    device=str(device),
                )
                sample, nfev = sampler()
            finally:
                hook.remove()
            elapsed = time.perf_counter() - run_start
            check_finite_tensor("ODE endpoint spectrum", sample)
            if int(nfev) != call_count[0]:
                raise RuntimeError(
                    f"SciPy nfev={nfev} != score DNN forward-hook count={call_count[0]} "
                    f"for {row['filename']} at tolerance {tolerance:g}"
                )

            with torch.inference_mode():
                waveform_out = model.to_audio(sample.squeeze(), length=int(waveform.size)).detach()
                waveform_out = waveform_out * norm_factor
            check_finite_tensor("ODE endpoint waveform", waveform_out)
            waveform_unclipped = waveform_out.squeeze().cpu().numpy().astype(np.float32, copy=False)
            if waveform_unclipped.shape != waveform.shape:
                raise ValueError(f"Output length mismatch for {row['filename']}: {waveform_unclipped.shape} vs {waveform.shape}")
            with torch.inference_mode():
                reconstructed_stft = model._stft(waveform_out.reshape(1, -1)).detach()
            check_finite_tensor("reconstructed waveform complex STFT", reconstructed_stft)
            # Match joint_enhance_by_dir.py's deployment WAV convention. The
            # convergence deltas below still use the finite, unclipped arrays
            # kept in memory, so output clipping cannot hide solver error.
            clipped = np.clip(waveform_unclipped, -1.0, 1.0)
            wav_path = tolerance_dirs[tolerance] / row["filename"]
            sf.write(wav_path, clipped, SAMPLE_RATE, subtype="FLOAT")
            endpoint_specs[tolerance] = sample.detach().cpu()
            endpoint_waves[tolerance] = waveform_unclipped.copy()
            endpoint_stfts[tolerance] = reconstructed_stft.cpu()
            rows_for_utt.append({
                "tolerance": tolerance,
                "rtol": tolerance,
                "atol": tolerance,
                "method": "RK45",
                "denoise": False,
                "t_start": float(model.sde.T),
                "t_end": float(model.t_eps),
                "nfev": int(nfev),
                "score_dnn_forward_calls": call_count[0],
                "nfev_matches_hook_count": True,
                "elapsed_seconds": elapsed,
                "endpoint_finite": True,
                "wav_path": str(wav_path.relative_to(ROOT)),
                "wav_sha256": sha256_file(wav_path),
                "wav_subtype": "FLOAT",
                "output_samples": int(waveform_unclipped.size),
                "unclipped_absmax": float(np.max(np.abs(waveform_unclipped))),
                "clipped_sample_count": int(np.count_nonzero(clipped != waveform_unclipped)),
                "convergence_deltas_use_unclipped_waveform": True,
                "shared_xT_sha256": xT_hash,
            })
            print(
                f"[ODE smoke] done  {row['filename']} rtol=atol={tolerance:.0e} "
                f"nfev={nfev} score_calls={call_count[0]} elapsed={elapsed:.2f}s",
                flush=True,
            )

        # The strictest integration is the rtol=atol=1e-7 reference. Track
        # convergence in both the internal transformed spectrum and the actual
        # complex STFT reanalyzed from each unclipped reconstructed waveform.
        for run in rows_for_utt:
            run["endpoint_spec_delta_to_rtol_1e-7"] = output_delta(
                endpoint_specs[1e-7], endpoint_specs[run["tolerance"]]
            )
            delta_wave = endpoint_waves[run["tolerance"]].astype(np.float64) - endpoint_waves[1e-7].astype(np.float64)
            denom = np.linalg.norm(endpoint_waves[1e-7].astype(np.float64).ravel())
            wave_delta = {
                "max_abs": float(np.max(np.abs(delta_wave))),
                "relative_l2": float(np.linalg.norm(delta_wave.ravel()) / max(denom, 1e-30)),
            }
            stft_delta = output_delta(endpoint_stfts[1e-7], endpoint_stfts[run["tolerance"]])
            run["endpoint_wave_delta_to_rtol_1e-7"] = wave_delta
            run["reconstructed_complex_stft_delta_to_rtol_1e-7"] = stft_delta
            run["wave_and_complex_stft_within_1e-3_of_rtol_1e-7"] = bool(
                wave_delta["relative_l2"] < 1e-3 and stft_delta["relative_l2"] < 1e-3
            )
        manifest_info.append({
            "filename": row["filename"],
            "uttid": row["uttid"],
            "noisy_path": row["noisy_path"],
            "noisy_sha256": row["noisy_sha256"],
            "sample_rate": SAMPLE_RATE,
            "input_samples": int(waveform.size),
            "shared_xT_path": str(xT_path.relative_to(ROOT)),
            "shared_xT_sha256": xT_hash,
            "prior_seed": SEED_BASE + index,
            "norm_factor_from_noisy_peak": norm_factor,
            "runs": rows_for_utt,
        })
    write_json(OUT / "metrics.json", {
        "experiment_id": "storm_error_attribution_p7_20261017_ode_convergence_smoke_20261017",
        "status": "completed",
        "scope": "four-utterance probability-flow ODE tolerance smoke; no quality metrics",
        "checkpoint": str(CHECKPOINT),
        "checkpoint_sha256": EXPECTED_CHECKPOINT_SHA256,
        "manifest": str(MANIFEST),
        "manifest_sha256": manifest_hash,
        "manifest_rows": manifest_info,
        "loaded_parameters": {name: actual for name, (actual, _expected) in loaded_parameters.items()},
        "model_load_seconds": model_config_seconds,
        "frontend": "StoRM, checkpoint-loaded with strict=True; model.eval(no_ema=False)",
        "ema_note": "P7 checkpoint EMA tracks the Router only; this smoke leaves RRSD routing disabled.",
        "sde": {"name": "OUVESDE", "saved_sampler_type": "pc", "N": 30, "theta": 1.5, "sigma_min": 0.05, "sigma_max": 0.5, "t_eps": 0.03},
        "spec_transform": {"n_fft": 510, "hop_length": 128, "num_frames": 256, "window": "periodic Hann", "transform_type": "exponent", "spec_abs_exponent": 0.5, "spec_factor": 0.15, "normalize": "noisy"},
        "sampler": {"implementation": "sgmse.sampling.get_ode_sampler", "probability_flow": True, "solver": "SciPy RK45", "denoise": False, "initial_state": "one saved OUVE prior x_T per utterance, reused across tolerances", "tolerances_rtol_atol": list(TOLERANCES)},
        "runtime": {"python": platform.python_version(), "torch": torch.__version__, "numpy": np.__version__, "scipy": __import__("scipy").__version__, "device": str(device), "elapsed_seconds": time.perf_counter() - start},
        "training": False,
        "quality_evaluation": False,
        "source_data_or_checkpoint_modified": False,
    })
    print(f"Wrote {OUT / 'metrics.json'}")


if __name__ == "__main__":
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    main()
