from __future__ import annotations

from typing import Any, Dict, Optional, Sequence

import torch
import torch.nn as nn

from sgmse.backbones import BackboneRegistry
from .base import BaseFrontend, ensure_mono_waveform, match_length, strip_state_dict_prefixes
from .registry import FrontendRegistry


def _get_window(window_type: str, window_length: int) -> torch.Tensor:
    if window_type == "sqrthann":
        return torch.sqrt(torch.hann_window(window_length, periodic=True))
    if window_type == "hann":
        return torch.hann_window(window_length, periodic=True)
    raise NotImplementedError(f"Window type {window_type} not implemented")




def _pad_spec(spec: torch.Tensor) -> torch.Tensor:
    t = spec.size(-1)
    num_pad = 0 if t % 64 == 0 else 64 - t % 64
    if num_pad == 0:
        return spec
    return torch.nn.functional.pad(spec, (0, num_pad, 0, 0), mode="constant", value=0.0)


def _extract_state_dict(ckpt: Any) -> Dict[str, Any]:
    if isinstance(ckpt, dict):
        if "state_dict" in ckpt and isinstance(ckpt["state_dict"], dict):
            return ckpt["state_dict"]
        for key in ("model", "generator", "net"):
            if key in ckpt and isinstance(ckpt[key], dict):
                return ckpt[key]
        return ckpt
    raise RuntimeError("Cannot parse checkpoint into a state_dict")


@FrontendRegistry.register("storm")
class StormFrontend(BaseFrontend):
    """Self-contained StoRM-style lighter NCSN++ waveform frontend.

    Notes
    -----
    This module keeps the *backbone family* aligned with StoRM's lighter NCSN++
    setting (by default nf=128, ch_mult=(1,2,2,2), num_res_blocks=1,
    attn_resolutions=(0,)). It is implemented entirely inside the current repo
    so it can be pre-trained with ``pretrain_frontend.py`` and then plugged into
    ``joint_train.py`` via ``--frontend storm``.

    The module performs a direct noisy->enhanced mapping in the STFT domain while
    reusing the existing ``ncsnpp`` backbone. To avoid invasive surgery on the
    original score backbone, the noisy spectrogram is duplicated into the
    backbone's two complex input slots.
    """

    @staticmethod
    def add_argparse_args(parser):
        parser.add_argument("--storm_nf", type=int, default=128)
        parser.add_argument("--storm_ch_mult", type=int, nargs='+', default=[1, 2, 2, 2])
        parser.add_argument("--storm_num_res_blocks", type=int, default=1)
        parser.add_argument("--storm_attn_resolutions", type=int, nargs='+', default=[0])
        parser.add_argument("--storm_time_cond", type=float, default=1.0)
        return parser

    def __init__(
        self,
        frontend_ckpt: Optional[str] = None,
        n_fft: int = 510,
        hop_length: int = 128,
        window: str = "hann",
        transform_type: str = "exponent",
        spec_factor: float = 0.15,
        spec_abs_exponent: float = 0.5,
        storm_nf: int = 128,
        storm_ch_mult: Sequence[int] = (1, 2, 2, 2),
        storm_num_res_blocks: int = 1,
        storm_attn_resolutions: Sequence[int] = (0,),
        storm_time_cond: float = 1.0,
        **kwargs,
    ):
        super().__init__()
        self.n_fft = int(n_fft)
        self.hop_length = int(hop_length)
        self.transform_type = str(transform_type)
        self.spec_factor = float(spec_factor)
        self.spec_abs_exponent = float(spec_abs_exponent)
        self.storm_time_cond = float(storm_time_cond)

        self.register_buffer("window", _get_window(window, self.n_fft), persistent=False)

        backbone_cls = BackboneRegistry.get_by_name("ncsnpp")
        self.backbone = backbone_cls(
            scale_by_sigma=False,
            conditional=False,
            nf=int(storm_nf),
            ch_mult=tuple(int(v) for v in storm_ch_mult),
            num_res_blocks=int(storm_num_res_blocks),
            attn_resolutions=tuple(int(v) for v in storm_attn_resolutions),
            centered=True,
        )

        if frontend_ckpt:
            ckpt = torch.load(frontend_ckpt, map_location="cpu", weights_only=False)
            state = _extract_state_dict(ckpt)
            filtered = {}
            for k, v in state.items():
                if k.startswith("frontend."):
                    filtered[k[len("frontend."):]] = v
                else:
                    filtered[k] = v
            filtered = strip_state_dict_prefixes(filtered)
            missing, unexpected = self.load_state_dict(filtered, strict=False)
            if missing or unexpected:
                print(
                    f"[StormFrontend] load_state_dict strict=False: missing={len(missing)} unexpected={len(unexpected)}",
                    flush=True,
                )

    def _spec_fwd(self, spec: torch.Tensor) -> torch.Tensor:
        if self.transform_type == "exponent":
            if self.spec_abs_exponent != 1:
                eps = 1e-8
                mag = torch.sqrt(spec.real * spec.real + spec.imag * spec.imag + eps)
                phase = spec / mag.clamp(min=1e-8)
                spec = mag.pow(self.spec_abs_exponent) * phase
                if (not torch.isfinite(spec.real).all()) or (not torch.isfinite(spec.imag).all()):
                    real = torch.nan_to_num(spec.real, nan=0.0, posinf=0.0, neginf=0.0)
                    imag = torch.nan_to_num(spec.imag, nan=0.0, posinf=0.0, neginf=0.0)
                    spec = torch.complex(real, imag)
            spec = spec * self.spec_factor
        elif self.transform_type == "log":
            spec = torch.log(1 + spec.abs()) * torch.exp(1j * spec.angle())
            spec = spec * self.spec_factor
        elif self.transform_type == "none":
            pass
        else:
            raise NotImplementedError(f"Unsupported transform_type={self.transform_type}")
        return spec

    def _spec_back(self, spec: torch.Tensor) -> torch.Tensor:
        if self.transform_type == "exponent":
            spec = spec / self.spec_factor
            if self.spec_abs_exponent != 1:
                spec = spec.abs() ** (1 / self.spec_abs_exponent) * torch.exp(1j * spec.angle())
        elif self.transform_type == "log":
            spec = spec / self.spec_factor
            spec = (torch.exp(spec.abs()) - 1) * torch.exp(1j * spec.angle())
        elif self.transform_type == "none":
            pass
        else:
            raise NotImplementedError(f"Unsupported transform_type={self.transform_type}")
        return spec

    def _stft(self, sig: torch.Tensor) -> torch.Tensor:
        if sig.dim() == 3 and sig.size(1) == 1:
            sig = sig[:, 0, :]
        window = self.window.to(sig.device)
        return torch.stft(
            sig,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.n_fft,
            window=window,
            center=True,
            return_complex=True,
        )

    def _istft(self, spec: torch.Tensor, length: Optional[int] = None) -> torch.Tensor:
        if spec.dim() == 4 and spec.size(1) == 1:
            spec = spec[:, 0, :, :]
        window = self.window.to(spec.device)
        return torch.istft(
            spec,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.n_fft,
            window=window,
            center=True,
            length=length,
        )

    def forward_waveform(self, noisy_wav: torch.Tensor) -> torch.Tensor:
        noisy_wav = ensure_mono_waveform(noisy_wav)
        target_len = noisy_wav.size(-1)
        spec = self._spec_fwd(self._stft(noisy_wav))
        orig_frames = spec.size(-1)
        spec = _pad_spec(spec.unsqueeze(1))  # [B, 1, F, Tpad]

        # Reuse the original NCSN++ complex-input backbone without modifying its U-Net core.
        x_pair = torch.cat([spec, spec], dim=1)  # [B, 2, F, Tpad] complex
        t = torch.full((spec.size(0),), self.storm_time_cond, device=spec.device, dtype=spec.real.dtype)
        pred = self.backbone(x_pair, t)[:, :, :, :orig_frames]
        pred = self._spec_back(pred)
        wav = self._istft(pred, length=target_len)
        wav = ensure_mono_waveform(wav)
        return match_length(wav, noisy_wav)
