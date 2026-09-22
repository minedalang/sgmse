"""Oracle-distilled reliability routing for diffusion speech enhancement.

RRSD learns a local, time-dependent convex routing coefficient alpha in [0, 1]
that interpolates the diffusion model's current x0 estimate with the predictive
frontend solution:

    x0_route = x0_base + alpha * (x0_front - x0_base)

The training target alpha* is a clean-referenced local projection oracle.  At
inference clean speech is not used: a small reliability router Q_phi predicts
alpha from quantities that are all available during reverse diffusion.
"""

from __future__ import annotations

from typing import Dict, Tuple
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class BandTimeReliabilityRouter(nn.Module):
    """Predict alpha(t,f,tau) on a low-resolution band-time grid.

    Unlike Anchor V2, this controller predicts a *convex frontend routing
    coefficient* rather than small signed residual coefficients.  No manual
    diffusion-time envelope is imposed: the oracle-distilled controller learns
    when frontend fallback is useful from data.
    """

    router_type = "oracle_distilled_band_time_reliability"

    def __init__(
        self,
        hidden_ch: int = 48,
        num_bands: int = 32,
        num_time_blocks: int = 16,
        init_alpha: float = 0.05,
    ) -> None:
        super().__init__()
        self.hidden_ch = int(hidden_ch)
        self.num_bands = max(int(num_bands), 1)
        self.num_time_blocks = max(int(num_time_blocks), 1)
        self.init_alpha = float(init_alpha)
        if not 0.0 < self.init_alpha < 1.0:
            raise ValueError("init_alpha must be in (0,1).")

        # Five local magnitude-stat maps x (mean/std/max), three local cosine
        # relations, and one normalized diffusion-time feature.
        in_ch = 5 * 3 + 3 + 1
        self.feature_proj = nn.Conv2d(in_ch, self.hidden_ch, kernel_size=1)
        self.context1 = nn.Conv2d(self.hidden_ch, self.hidden_ch, kernel_size=3, padding=1)
        self.context2 = nn.Conv2d(self.hidden_ch, self.hidden_ch, kernel_size=3, padding=1)
        self.output = nn.Conv2d(self.hidden_ch, 1, kernel_size=1)

        # Start close to a no-op instead of alpha=0.5.
        nn.init.zeros_(self.output.weight)
        init_logit = math.log(self.init_alpha / (1.0 - self.init_alpha))
        nn.init.constant_(self.output.bias, init_logit)

    @staticmethod
    def _real_dtype(x: torch.Tensor) -> torch.dtype:
        return x.real.dtype if torch.is_complex(x) else x.dtype

    @staticmethod
    def _pool_size(x: torch.Tensor, num_bands: int, num_time_blocks: int) -> Tuple[int, int]:
        if x.ndim != 4:
            raise ValueError(f"RRSD expects [B,C,F,T], got {tuple(x.shape)}")
        return (
            max(1, min(int(num_bands), int(x.shape[-2]))),
            max(1, min(int(num_time_blocks), int(x.shape[-1]))),
        )

    @classmethod
    def _tile_stats(cls, x: torch.Tensor, out_size: Tuple[int, int]) -> torch.Tensor:
        mag = torch.abs(x).float() if torch.is_complex(x) else x.abs().float()
        mag = mag.mean(dim=1, keepdim=True)
        mean = F.adaptive_avg_pool2d(mag, out_size)
        mean_sq = F.adaptive_avg_pool2d(mag.square(), out_size)
        std = (mean_sq - mean.square()).clamp_min(0.0).sqrt()
        maxv = F.adaptive_max_pool2d(mag, out_size)
        return torch.cat([mean, std, maxv], dim=1)

    @staticmethod
    def _real_inner_map(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        if torch.is_complex(a) or torch.is_complex(b):
            v = torch.real(torch.conj(a) * b)
        else:
            v = a * b
        return v.float().sum(dim=1, keepdim=True)

    @staticmethod
    def _energy_map(x: torch.Tensor) -> torch.Tensor:
        return torch.abs(x).float().square().sum(dim=1, keepdim=True)

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

    def forward(
        self,
        *,
        x_t: torch.Tensor,
        x0_base: torch.Tensor,
        x0_front: torch.Tensor,
        y_main: torch.Tensor,
        y_cond: torch.Tensor,
        t: torch.Tensor,
        t_max: float = 1.0,
        detach_inputs: bool = True,
        scale: float = 1.0,
    ) -> Dict[str, torch.Tensor]:
        if detach_inputs:
            x_t = x_t.detach()
            x0_base = x0_base.detach()
            x0_front = x0_front.detach()
            y_main = y_main.detach()
            y_cond = y_cond.detach()
            t = t.detach()

        if t.ndim == 0:
            t = torch.full(
                (x_t.shape[0],), float(t.item()), device=x_t.device, dtype=torch.float32
            )
        else:
            t = t.to(device=x_t.device, dtype=torch.float32).flatten()

        out_size = self._pool_size(x0_base, self.num_bands, self.num_time_blocks)

        # The most important evidence is the predictive-generative disagreement.
        d_front = x0_front - x0_base
        d_main = y_main - x0_base
        d_cond = y_cond - x0_base
        state_gap = x_t - x0_base

        feat = torch.cat(
            [
                self._tile_stats(x0_base, out_size),
                self._tile_stats(d_front, out_size),
                self._tile_stats(d_main, out_size),
                self._tile_stats(d_cond, out_size),
                self._tile_stats(state_gap, out_size),
                self._tile_cosine(d_front, d_cond, out_size),
                self._tile_cosine(d_front, d_main, out_size),
                self._tile_cosine(d_main, d_cond, out_size),
            ],
            dim=1,
        )
        t_feat = (t / max(float(t_max), 1e-8)).clamp(0.0, 1.0)
        t_feat = t_feat.view(-1, 1, 1, 1).expand(-1, 1, out_size[0], out_size[1])
        feat = torch.cat([feat, t_feat], dim=1)

        h = F.silu(self.feature_proj(feat))
        h = F.silu(self.context1(h))
        h = F.silu(self.context2(h))
        alpha_tile_raw = torch.sigmoid(self.output(h))
        scale = max(float(scale), 0.0)
        alpha_tile = (alpha_tile_raw * scale).clamp(0.0, 1.0)

        full_size = (int(x0_base.shape[-2]), int(x0_base.shape[-1]))
        alpha_map = F.interpolate(alpha_tile, size=full_size, mode="bilinear", align_corners=False)
        alpha_map = alpha_map.to(dtype=self._real_dtype(x0_base))

        delta_x0 = alpha_map * (x0_front - x0_base)
        x0_route = x0_base + delta_x0
        return {
            "alpha_tile": alpha_tile,
            "alpha_tile_raw": alpha_tile_raw,
            "alpha_map": alpha_map,
            "delta_x0": delta_x0,
            "x0_route": x0_route,
        }


def _tile_real_inner(a: torch.Tensor, b: torch.Tensor, out_size: Tuple[int, int]) -> torch.Tensor:
    if torch.is_complex(a) or torch.is_complex(b):
        prod = torch.real(torch.conj(a) * b)
    else:
        prod = a * b
    return F.adaptive_avg_pool2d(prod.float().sum(dim=1, keepdim=True), out_size)


def _tile_energy(x: torch.Tensor, out_size: Tuple[int, int]) -> torch.Tensor:
    energy = torch.abs(x).float().square().sum(dim=1, keepdim=True)
    return F.adaptive_avg_pool2d(energy, out_size)


def solve_frontend_projection_oracle(
    *,
    target: torch.Tensor,
    x0_base: torch.Tensor,
    x0_front: torch.Tensor,
    num_bands: int,
    num_time_blocks: int,
    eps: float = 1e-6,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """Local convex projection oracle used only during training.

    Finds alpha* in [0,1] minimizing

        || target - [x0_base + alpha (x0_front-x0_base)] ||^2

    independently on each band-time tile.
    """
    out_size = BandTimeReliabilityRouter._pool_size(
        x0_base, num_bands=num_bands, num_time_blocks=num_time_blocks
    )
    d = x0_front - x0_base
    r = target - x0_base

    numerator = _tile_real_inner(d, r, out_size)
    denominator = _tile_energy(d, out_size).clamp_min(float(eps))
    alpha_raw = numerator / denominator
    alpha_star = alpha_raw.clamp(0.0, 1.0)

    base_error = _tile_energy(r, out_size)
    oracle_error = (
        base_error
        - 2.0 * alpha_star * numerator
        + alpha_star.square() * denominator
    ).clamp_min(0.0)
    benefit = (base_error - oracle_error).clamp_min(0.0)
    benefit_ratio = benefit / base_error.clamp_min(float(eps))

    stats = {
        "alpha_raw": alpha_raw,
        "base_error": base_error,
        "oracle_error": oracle_error,
        "benefit": benefit,
        "benefit_ratio": benefit_ratio,
        "disagreement_energy": denominator,
    }
    return alpha_star.detach(), {k: v.detach() for k, v in stats.items()}
