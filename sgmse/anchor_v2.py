"""Fine-grained band-time conditional evidence calibration for OUVE sampling.

Anchor V2 does not replace the score network.  It estimates only the part of
its current x0 error that can be explained by two observed condition directions:

    u_pre   = Y_pre - x0_base
    u_noisy = Y_noisy - Y_pre

The controller predicts signed, bounded coefficients on a low-resolution
frequency-band x time-block grid and expands them smoothly to the STFT grid:

    x0_calib = x0_base + a * u_pre + b * u_noisy

The resulting x0 residual is converted back into a score residual by the joint
model.  The old anchor implementation is intentionally left untouched.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class BandTimeAnchorV2Controller(nn.Module):
    """Predict fine-grained signed gates for conditional x0 residuals.

    The controller operates on an adaptive ``num_bands x num_time_blocks``
    grid.  Its output layer is zero-initialized, so enabling the branch starts
    from an exact no-op and lets training introduce corrections gradually.
    """

    controller_type = "band_time_x0_residual_v2"

    def __init__(
        self,
        hidden_ch: int = 48,
        num_bands: int = 32,
        num_time_blocks: int = 16,
        a_max: float = 0.25,
        b_max: float = 0.15,
        time_power: float = 1.0,
        infer_scale: float = 1.0,
    ) -> None:
        super().__init__()
        self.hidden_ch = int(hidden_ch)
        self.num_bands = max(int(num_bands), 1)
        self.num_time_blocks = max(int(num_time_blocks), 1)
        self.a_max = float(a_max)
        self.b_max = float(b_max)
        self.time_power = float(time_power)
        self.infer_scale = float(infer_scale)

        # Four maps x (mean/std/max) + three local complex correlations + time.
        in_ch = 4 * 3 + 3 + 1
        self.feature_proj = nn.Conv2d(in_ch, self.hidden_ch, kernel_size=1)
        self.context = nn.Conv2d(self.hidden_ch, self.hidden_ch, kernel_size=3, padding=1)
        self.output = nn.Conv2d(self.hidden_ch, 2, kernel_size=1)

        # Start from no calibration.  This is important when jointly training
        # from an existing score/frontend configuration.
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    @staticmethod
    def _real_dtype(x: torch.Tensor) -> torch.dtype:
        return x.real.dtype if torch.is_complex(x) else x.dtype

    @staticmethod
    def _magnitude(x: torch.Tensor) -> torch.Tensor:
        return torch.abs(x) if torch.is_complex(x) else x.abs()

    @staticmethod
    def _pool_size(x: torch.Tensor, num_bands: int, num_time_blocks: int) -> Tuple[int, int]:
        return (
            max(1, min(int(num_bands), int(x.shape[-2]))),
            max(1, min(int(num_time_blocks), int(x.shape[-1]))),
        )

    @classmethod
    def _tile_stats(cls, x: torch.Tensor, out_size: Tuple[int, int]) -> torch.Tensor:
        """Return local magnitude mean/std/max as [B,3,Nf,Nt]."""
        mag = cls._magnitude(x).float()
        # Collapse feature/channel dimension without collapsing time.
        if mag.dim() != 4:
            raise ValueError(f"Anchor V2 expects [B,C,F,T], got shape={tuple(mag.shape)}")
        mag = mag.mean(dim=1, keepdim=True)
        mean = F.adaptive_avg_pool2d(mag, out_size)
        mean_sq = F.adaptive_avg_pool2d(mag.square(), out_size)
        std = (mean_sq - mean.square()).clamp_min(0.0).sqrt()
        maxv = F.adaptive_max_pool2d(mag, out_size)
        return torch.cat([mean, std, maxv], dim=1)

    @staticmethod
    def _real_inner_map(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        if torch.is_complex(a) or torch.is_complex(b):
            prod = torch.real(torch.conj(a) * b)
        else:
            prod = a * b
        return prod.float().sum(dim=1, keepdim=True)

    @staticmethod
    def _energy_map(a: torch.Tensor) -> torch.Tensor:
        return torch.abs(a).float().square().sum(dim=1, keepdim=True)

    @classmethod
    def _tile_cosine(
        cls,
        a: torch.Tensor,
        b: torch.Tensor,
        out_size: Tuple[int, int],
        eps: float = 1e-8,
    ) -> torch.Tensor:
        inner = F.adaptive_avg_pool2d(cls._real_inner_map(a, b), out_size)
        ea = F.adaptive_avg_pool2d(cls._energy_map(a), out_size)
        eb = F.adaptive_avg_pool2d(cls._energy_map(b), out_size)
        return (inner / (ea.mul(eb).clamp_min(eps).sqrt())).clamp(-1.0, 1.0)

    def _time_envelope(self, t: torch.Tensor, t_max: float = 1.0) -> torch.Tensor:
        t_norm = (t.float() / max(float(t_max), 1e-8)).clamp(0.0, 1.0)
        s = t_norm.pow(float(self.time_power))
        return (4.0 * s * (1.0 - s)).clamp(0.0, 1.0)

    def forward(
        self,
        x_t: torch.Tensor,
        x0_base: torch.Tensor,
        y_pre: torch.Tensor,
        y_noisy: torch.Tensor,
        t: torch.Tensor,
        t_max: float = 1.0,
        detach_inputs: bool = True,
        a_max_override: Optional[float] = None,
        b_max_override: Optional[float] = None,
        scale: Optional[float] = None,
    ) -> Dict[str, torch.Tensor]:
        if detach_inputs:
            x_t = x_t.detach()
            x0_base = x0_base.detach()
            y_pre = y_pre.detach()
            y_noisy = y_noisy.detach()
            t = t.detach()

        if t.dim() == 0:
            t = torch.full(
                (x_t.shape[0],), float(t.item()), device=x_t.device, dtype=torch.float32
            )
        else:
            t = t.to(device=x_t.device, dtype=torch.float32).flatten()

        out_size = self._pool_size(x0_base, self.num_bands, self.num_time_blocks)

        # Two semantically distinct condition directions.
        u_pre = y_pre - x0_base
        u_noisy = y_noisy - y_pre
        state_gap = x_t - x0_base

        feat = torch.cat(
            [
                self._tile_stats(u_pre, out_size),
                self._tile_stats(u_noisy, out_size),
                self._tile_stats(x0_base, out_size),
                self._tile_stats(state_gap, out_size),
                self._tile_cosine(u_pre, u_noisy, out_size),
                self._tile_cosine(u_pre, x0_base, out_size),
                self._tile_cosine(u_noisy, y_pre, out_size),
            ],
            dim=1,
        )
        t_feat = (t / max(float(t_max), 1e-8)).clamp(0.0, 1.0)
        t_feat = t_feat.view(-1, 1, 1, 1).expand(-1, 1, out_size[0], out_size[1])
        feat = torch.cat([feat, t_feat], dim=1)

        h = F.silu(self.feature_proj(feat))
        h = F.silu(self.context(h))
        raw = self.output(h)

        envelope = self._time_envelope(t, t_max=t_max).view(-1, 1, 1, 1)
        scale_eff = max(float(self.infer_scale if scale is None else scale), 0.0)
        a_cap_value = max(float(self.a_max if a_max_override is None else a_max_override) * scale_eff, 0.0)
        b_cap_value = max(float(self.b_max if b_max_override is None else b_max_override) * scale_eff, 0.0)

        a_tile = torch.tanh(raw[:, 0:1]) * (a_cap_value * envelope)
        b_tile = torch.tanh(raw[:, 1:2]) * (b_cap_value * envelope)

        full_size = (int(x0_base.shape[-2]), int(x0_base.shape[-1]))
        a_map = F.interpolate(a_tile, size=full_size, mode="bilinear", align_corners=False)
        b_map = F.interpolate(b_tile, size=full_size, mode="bilinear", align_corners=False)
        out_dtype = self._real_dtype(x0_base)
        a_map = a_map.to(dtype=out_dtype)
        b_map = b_map.to(dtype=out_dtype)

        delta_x0_pre = a_map * u_pre
        delta_x0_noisy = b_map * u_noisy
        delta_x0 = delta_x0_pre + delta_x0_noisy
        x0_calib = x0_base + delta_x0

        return {
            "x0_calib": x0_calib,
            "delta_x0": delta_x0,
            "delta_x0_pre": delta_x0_pre,
            "delta_x0_noisy": delta_x0_noisy,
            "u_pre": u_pre,
            "u_noisy": u_noisy,
            "a_map": a_map,
            "b_map": b_map,
            "a_tile": a_tile,
            "b_tile": b_tile,
            "E_t": envelope,
            "a_cap": torch.as_tensor(a_cap_value, device=x_t.device, dtype=torch.float32) * envelope,
            "b_cap": torch.as_tensor(b_cap_value, device=x_t.device, dtype=torch.float32) * envelope,
        }


def _tile_real_inner(a: torch.Tensor, b: torch.Tensor, out_size: Tuple[int, int]) -> torch.Tensor:
    if torch.is_complex(a) or torch.is_complex(b):
        prod = torch.real(torch.conj(a) * b)
    else:
        prod = a * b
    prod = prod.float().sum(dim=1, keepdim=True)
    return F.adaptive_avg_pool2d(prod, out_size)


def _tile_energy(a: torch.Tensor, out_size: Tuple[int, int]) -> torch.Tensor:
    energy = torch.abs(a).float().square().sum(dim=1, keepdim=True)
    return F.adaptive_avg_pool2d(energy, out_size)


def solve_two_direction_tile_oracle(
    residual: torch.Tensor,
    u_pre: torch.Tensor,
    u_noisy: torch.Tensor,
    num_bands: int,
    num_time_blocks: int,
    eps: float = 1e-6,
) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
    """Solve a local 2-D least-squares projection on the band-time grid.

    Finds ``a*, b*`` minimizing ``||residual - a*u_pre - b*u_noisy||^2``
    independently in each adaptive tile.  A scale-aware ridge keeps nearly
    collinear or silent tiles numerically stable.
    """
    out_size = BandTimeAnchorV2Controller._pool_size(residual, num_bands, num_time_blocks)

    g11 = _tile_energy(u_pre, out_size)
    g22 = _tile_energy(u_noisy, out_size)
    g12 = _tile_real_inner(u_pre, u_noisy, out_size)
    q1 = _tile_real_inner(u_pre, residual, out_size)
    q2 = _tile_real_inner(u_noisy, residual, out_size)

    ridge = float(eps) * (g11 + g22 + 1.0)
    h11 = g11 + ridge
    h22 = g22 + ridge
    det = (h11 * h22 - g12.square()).clamp_min(float(eps))

    a_star = (q1 * h22 - q2 * g12) / det
    b_star = (q2 * h11 - q1 * g12) / det

    # Useful diagnostics: 1 means strongly identifiable, 0 means near-collinear.
    conditioning = (det / (h11 * h22).clamp_min(float(eps))).clamp(0.0, 1.0)
    stats = {
        "conditioning": conditioning,
        "u_pre_energy": g11,
        "u_noisy_energy": g22,
    }
    return a_star, b_star, stats
