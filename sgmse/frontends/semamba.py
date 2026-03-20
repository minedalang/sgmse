from __future__ import annotations

import importlib
import sys
import types
from pathlib import Path
from typing import Any, Dict, Optional

import torch

from .base import BaseFrontend, ensure_mono_waveform, match_length


def _load_yaml(path: str) -> Dict[str, Any]:
    import yaml
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _import_semamba_package(models_dir: Path, package_name: str = "_semamba_pkg"):
    if package_name not in sys.modules:
        pkg = types.ModuleType(package_name)
        pkg.__path__ = [str(models_dir)]
        sys.modules[package_name] = pkg
    if str(models_dir) not in sys.path:
        sys.path.insert(0, str(models_dir))
    return package_name


class SemambaFrontend(BaseFrontend):
    @staticmethod
    def add_argparse_args(parser):
        return parser

    def __init__(
        self,
        frontend_root: Optional[str] = None,
        frontend_config: Optional[str] = None,
        frontend_ckpt: Optional[str] = None,
        **kwargs,
    ):
        super().__init__()
        if not frontend_root:
            raise ValueError("frontend=semamba requires --frontend_root")
        if not frontend_config:
            raise ValueError("frontend=semamba requires --frontend_config")

        self.frontend_root = str(frontend_root)
        self.cfg = _load_yaml(frontend_config)

        models_dir = Path(self.frontend_root) / "models"
        if not models_dir.exists():
            raise FileNotFoundError(f"SEMamba models dir not found: {models_dir}")

        pkg = _import_semamba_package(models_dir)
        gen = importlib.import_module(f"{pkg}.generator")
        stfts = importlib.import_module(f"{pkg}.stfts")

        self.model = gen.SEMamba(self.cfg)
        self._sem_stft = stfts.mag_phase_stft
        self._sem_istft = stfts.mag_phase_istft

        if frontend_ckpt:
            ckpt = torch.load(frontend_ckpt, map_location="cpu", weights_only=False)
            if isinstance(ckpt, dict) and "state_dict" in ckpt:
                state = ckpt["state_dict"]
            elif isinstance(ckpt, dict) and "generator" in ckpt:
                state = ckpt["generator"]
            elif isinstance(ckpt, dict) and "model" in ckpt:
                state = ckpt["model"]
            else:
                state = ckpt
            missing, unexpected = self.model.load_state_dict(state, strict=False)
            if missing or unexpected:
                print(
                    f"[SemambaFrontend] load_state_dict strict=False: missing={len(missing)} unexpected={len(unexpected)}",
                    flush=True,
                )

        stft_cfg = self.cfg.get("stft_cfg", {})
        model_cfg = self.cfg.get("model_cfg", {})
        self.n_fft = int(stft_cfg.get("n_fft", 400))
        self.hop = int(stft_cfg.get("hop_size", 100))
        self.win = int(stft_cfg.get("win_size", 400))
        self.cf = float(model_cfg.get("compress_factor", 1.0))

    def forward_waveform(self, noisy_wav: torch.Tensor) -> torch.Tensor:
        noisy_wav = ensure_mono_waveform(noisy_wav)
        y = noisy_wav.squeeze(1)
        noisy_mag, noisy_pha, _ = self._sem_stft(
            y,
            n_fft=self.n_fft,
            hop_size=self.hop,
            win_size=self.win,
            compress_factor=self.cf,
            center=True,
            addeps=False,
        )
        den_mag, den_pha, _ = self.model(noisy_mag, noisy_pha)
        y_pre = self._sem_istft(
            den_mag,
            den_pha,
            n_fft=self.n_fft,
            hop_size=self.hop,
            win_size=self.win,
            compress_factor=self.cf,
            center=True,
        )
        y_pre = y_pre.unsqueeze(1)
        return match_length(y_pre, noisy_wav)
