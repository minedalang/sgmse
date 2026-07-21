"""Frontend-guided band-time variance path simulation (Version A).

The independent estimator predicts only a band-time log variance ratio

    log_var_ratio = log(rho^2)

for the standardized reverse-trajectory residual.  It never predicts clean
speech, score, or a mean shift.  A zero output exactly recovers the original
analytical forward marginal.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass(frozen=True)
class PathSimVAConfig:
    hidden_ch: int = 48
    num_bands: int = 32
    num_time_blocks: int = 16
    min_rho: float = 0.5
    max_rho: float = 2.0

    def as_dict(self) -> Dict[str, float]:
        return {
            "hidden_ch": int(self.hidden_ch),
            "num_bands": int(self.num_bands),
            "num_time_blocks": int(self.num_time_blocks),
            "min_rho": float(self.min_rho),
            "max_rho": float(self.max_rho),
        }


class BandTimeVariancePathEstimator(nn.Module):
    """Predict a low-resolution conditional log variance-ratio map.

    Inputs are derived only from the frontend output and the frontend-removed
    residual.  The estimator therefore remains a trajectory-distribution model
    rather than a second denoiser.
    """

    model_type = "frontend_guided_band_time_variance_pathsim_vA"

    def __init__(
        self,
        hidden_ch: int = 48,
        num_bands: int = 32,
        num_time_blocks: int = 16,
        min_rho: float = 0.5,
        max_rho: float = 2.0,
    ) -> None:
        super().__init__()
        if min_rho <= 0.0 or max_rho <= 0.0 or min_rho > max_rho:
            raise ValueError(f"Invalid rho range: [{min_rho}, {max_rho}]")
        self.hidden_ch = int(hidden_ch)
        self.num_bands = max(1, int(num_bands))
        self.num_time_blocks = max(1, int(num_time_blocks))
        self.min_rho = float(min_rho)
        self.max_rho = float(max_rho)

        # pre: logmag mean/std/max; residual: logmag mean/std/max;
        # noisy/pre log-energy ratio; local coherence; normalized time.
        in_ch = 3 + 3 + 1 + 1 + 1
        self.in_proj = nn.Conv2d(in_ch, self.hidden_ch, kernel_size=1)
        self.block1 = nn.Conv2d(self.hidden_ch, self.hidden_ch, kernel_size=3, padding=1)
        self.block2 = nn.Conv2d(self.hidden_ch, self.hidden_ch, kernel_size=3, padding=1)
        self.out = nn.Conv2d(self.hidden_ch, 1, kernel_size=1)

        # Zero output => log rho^2 = 0 => rho = 1 => standard path.
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def get_config(self) -> Dict[str, float]:
        return PathSimVAConfig(
            hidden_ch=self.hidden_ch,
            num_bands=self.num_bands,
            num_time_blocks=self.num_time_blocks,
            min_rho=self.min_rho,
            max_rho=self.max_rho,
        ).as_dict()

    @staticmethod
    def _pool_size(x: torch.Tensor, num_bands: int, num_time_blocks: int) -> Tuple[int, int]:
        return (
            max(1, min(int(num_bands), int(x.shape[-2]))),
            max(1, min(int(num_time_blocks), int(x.shape[-1]))),
        )

    @staticmethod
    def _mag(x: torch.Tensor) -> torch.Tensor:
        return torch.abs(x).float()

    @classmethod
    def _logmag_stats(cls, x: torch.Tensor, out_size: Tuple[int, int]) -> torch.Tensor:
        if x.ndim != 4:
            raise ValueError(f"Expected [B,C,F,T], got {tuple(x.shape)}")
        logmag = torch.log1p(cls._mag(x)).mean(dim=1, keepdim=True)
        mean = F.adaptive_avg_pool2d(logmag, out_size)
        mean2 = F.adaptive_avg_pool2d(logmag.square(), out_size)
        std = (mean2 - mean.square()).clamp_min(0.0).sqrt()
        maxv = F.adaptive_max_pool2d(logmag, out_size)
        return torch.cat((mean, std, maxv), dim=1)

    @staticmethod
    def _energy(x: torch.Tensor) -> torch.Tensor:
        return torch.abs(x).float().square().sum(dim=1, keepdim=True)

    @classmethod
    def _coherence(cls, a: torch.Tensor, b: torch.Tensor, out_size: Tuple[int, int]) -> torch.Tensor:
        if torch.is_complex(a) or torch.is_complex(b):
            inner = torch.real(torch.conj(a) * b).float().sum(dim=1, keepdim=True)
        else:
            inner = (a * b).float().sum(dim=1, keepdim=True)
        inner = F.adaptive_avg_pool2d(inner, out_size)
        ea = F.adaptive_avg_pool2d(cls._energy(a), out_size)
        eb = F.adaptive_avg_pool2d(cls._energy(b), out_size)
        return (inner / (ea.mul(eb).clamp_min(1e-8).sqrt())).clamp(-1.0, 1.0)

    def _bound_log_var(self, raw: torch.Tensor) -> torch.Tensor:
        lo = 2.0 * math.log(self.min_rho)
        hi = 2.0 * math.log(self.max_rho)
        center = 0.5 * (lo + hi)
        half = 0.5 * (hi - lo)
        if half <= 0.0:
            return torch.full_like(raw, center)
        # Shift raw so raw=0 maps exactly to log-var=0 whenever 1 is inside
        # the configured rho interval.
        zero_normalized = (0.0 - center) / half
        zero_normalized = max(-0.999999, min(0.999999, zero_normalized))
        bias = math.atanh(zero_normalized)
        return center + half * torch.tanh(raw + bias)

    def forward(
        self,
        y_pre: torch.Tensor,
        y_noisy: torch.Tensor,
        t: torch.Tensor,
        t_max: float = 1.0,
        return_full: bool = False,
        full_size: Optional[Tuple[int, int]] = None,
    ) -> Dict[str, torch.Tensor]:
        if y_pre.shape != y_noisy.shape:
            raise ValueError(f"y_pre/y_noisy shape mismatch: {y_pre.shape} vs {y_noisy.shape}")
        if t.ndim == 0:
            t = t.expand(y_pre.shape[0])
        t = t.to(device=y_pre.device, dtype=torch.float32).flatten()

        out_size = self._pool_size(y_pre, self.num_bands, self.num_time_blocks)
        residual = y_noisy - y_pre
        pre_stats = self._logmag_stats(y_pre, out_size)
        residual_stats = self._logmag_stats(residual, out_size)

        e_pre = F.adaptive_avg_pool2d(self._energy(y_pre), out_size)
        e_res = F.adaptive_avg_pool2d(self._energy(residual), out_size)
        log_energy_ratio = torch.log((e_res + 1e-8) / (e_pre + 1e-8)).clamp(-8.0, 8.0)
        coherence = self._coherence(y_pre, y_noisy, out_size)
        t_map = (t / max(float(t_max), 1e-8)).clamp(0.0, 1.0)
        t_map = t_map.view(-1, 1, 1, 1).expand(-1, 1, out_size[0], out_size[1])

        feat = torch.cat((pre_stats, residual_stats, log_energy_ratio, coherence, t_map), dim=1)
        h = F.silu(self.in_proj(feat))
        h = F.silu(self.block1(h))
        h = F.silu(self.block2(h) + h)
        log_var_tile = self._bound_log_var(self.out(h))

        result = {
            "log_var_tile": log_var_tile,
            "rho_tile": torch.exp(0.5 * log_var_tile),
            "feature_energy_pre": e_pre,
            "feature_energy_residual": e_res,
        }
        if return_full:
            size = full_size or (int(y_pre.shape[-2]), int(y_pre.shape[-1]))
            log_var_full = F.interpolate(log_var_tile, size=size, mode="bilinear", align_corners=False)
            result["log_var_full"] = log_var_full
            result["rho_full"] = torch.exp(0.5 * log_var_full)
        return result


def standardized_rollout_residual(
    x_t_rollout: torch.Tensor,
    mean_prior: torch.Tensor,
    std_prior: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Return z=(x_rollout-mean_prior)/std_prior on a common scale."""
    std = std_prior.to(device=x_t_rollout.device, dtype=x_t_rollout.real.dtype)
    while std.ndim < x_t_rollout.ndim:
        std = std.unsqueeze(-1)
    return (x_t_rollout - mean_prior) / std.clamp_min(float(eps))


def tile_complex_energy(z: torch.Tensor, out_size: Tuple[int, int]) -> torch.Tensor:
    """Average |z|^2 over channels and samples inside each band-time tile."""
    energy = torch.abs(z).float().square().mean(dim=1, keepdim=True)
    return F.adaptive_avg_pool2d(energy, out_size)


def variance_nll_loss(
    log_var_tile: torch.Tensor,
    z_target: torch.Tensor,
    weight_tile: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """Zero-mean complex Gaussian NLL for band-time variance prediction.

    The optimal exp(log_var) equals E[|z|^2] for each conditional tile.
    """
    target_energy = tile_complex_energy(z_target, log_var_tile.shape[-2:])
    nll_map = torch.exp(-log_var_tile) * target_energy + log_var_tile
    if weight_tile is None:
        loss = nll_map.mean()
    else:
        w = weight_tile.detach().float().clamp_min(0.0)
        loss = (nll_map * w).sum() / w.sum().clamp_min(1.0)
    stats = {
        "target_energy_mean": target_energy.detach().mean(),
        "target_log_var_mean": torch.log(target_energy.clamp_min(1e-8)).detach().mean(),
        "pred_log_var_mean": log_var_tile.detach().mean(),
        "pred_rho_mean": torch.exp(0.5 * log_var_tile.detach()).mean(),
        "nll_mean": nll_map.detach().mean(),
    }
    return loss, stats


def standard_normal_kl(log_var_tile: torch.Tensor) -> torch.Tensor:
    """KL(N(0,exp(s)) || N(0,1)) for the variance-only model."""
    return 0.5 * (torch.exp(log_var_tile) - 1.0 - log_var_tile).mean()


def total_variation(log_var_tile: torch.Tensor) -> torch.Tensor:
    loss = torch.zeros((), device=log_var_tile.device, dtype=log_var_tile.dtype)
    if log_var_tile.shape[-2] > 1:
        loss = loss + log_var_tile.diff(dim=-2).abs().mean()
    if log_var_tile.shape[-1] > 1:
        loss = loss + log_var_tile.diff(dim=-1).abs().mean()
    return loss


def load_pathsim_va_checkpoint(path: str, map_location: str | torch.device = "cpu"):
    try:
        obj = torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        obj = torch.load(path, map_location=map_location)
    if not isinstance(obj, dict):
        raise ValueError("PathSim V-A checkpoint must be a dict")
    config = dict(obj.get("config", {}))
    state = obj.get("state_dict", obj)
    # Accept checkpoints saved from a wrapper with path_model.* keys.
    if any(k.startswith("path_model.") for k in state):
        state = {k[len("path_model."):]: v for k, v in state.items() if k.startswith("path_model.")}
    model = BandTimeVariancePathEstimator(**config)
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        raise RuntimeError(
            f"Invalid PathSim V-A checkpoint: missing={missing[:10]}, unexpected={unexpected[:10]}"
        )
    return model, obj
