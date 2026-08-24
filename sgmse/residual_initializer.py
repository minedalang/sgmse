"""Residual priors for residual-coordinate speech enhancement.

Two variants are provided for clean ablations:

1) ``direct``: predict the clean correction residual directly,
      R_target = X_clean - Y_pre.
2) ``confidence_gate``: predict only a real-valued confidence map G in [0,1]
   and form the prior from the observable residual
      R_obs   = Y_noisy - Y_pre,
      R_prior = G * R_obs.

The confidence-gated version is intentionally conservative: the small network
never has to synthesize a complex residual from scratch.  It only decides how
much of each local time-frequency residual observation should be trusted.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class _DilatedResidualBlock(nn.Module):
    def __init__(self, channels: int, dilation: int = 1, groups: int = 8):
        super().__init__()
        groups = max(1, min(int(groups), int(channels)))
        while channels % groups != 0 and groups > 1:
            groups -= 1
        pad = int(dilation)
        self.norm1 = nn.GroupNorm(groups, channels)
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=pad, dilation=dilation)
        self.norm2 = nn.GroupNorm(groups, channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=pad, dilation=dilation)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.conv1(F.silu(self.norm1(x)))
        h = self.conv2(F.silu(self.norm2(h)))
        return x + h


class _ResidualFeatureMixin:
    @staticmethod
    def _to_ri(x: torch.Tensor) -> torch.Tensor:
        if torch.is_complex(x):
            return torch.cat([x.real, x.imag], dim=1)
        if x.dim() != 4:
            raise ValueError(f"Expected [B,C,F,T], got {tuple(x.shape)}")
        if x.size(1) == 2:
            return x
        if x.size(1) == 1:
            return torch.cat([x, torch.zeros_like(x)], dim=1)
        raise ValueError(f"Cannot interpret non-complex tensor with {x.size(1)} channels")

    @staticmethod
    def _to_complex(x_ri: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
        if x_ri.size(1) != 2:
            raise ValueError(f"Expected two output channels, got {x_ri.size(1)}")
        if torch.is_complex(ref):
            return torch.complex(
                x_ri[:, 0:1].to(dtype=ref.real.dtype),
                x_ri[:, 1:2].to(dtype=ref.real.dtype),
            )
        return x_ri[:, 0:1].to(dtype=ref.dtype)

    @staticmethod
    def _build_features(y_pre: torch.Tensor, y_noisy: torch.Tensor) -> torch.Tensor:
        if y_pre.shape != y_noisy.shape:
            raise ValueError(
                f"y_pre and y_noisy must have identical shapes, got "
                f"{tuple(y_pre.shape)} and {tuple(y_noisy.shape)}"
            )
        residual_obs = y_noisy - y_pre
        return torch.cat(
            [
                _ResidualFeatureMixin._to_ri(y_pre),
                _ResidualFeatureMixin._to_ri(residual_obs),
            ],
            dim=1,
        ).float()


class DirectResidualInitPredictor(nn.Module, _ResidualFeatureMixin):
    """Legacy direct predictor: H(Y_pre,Y_noisy) -> complex R_target estimate."""

    def __init__(
        self,
        hidden_ch: int = 48,
        num_blocks: int = 4,
        max_dilation: int = 8,
    ) -> None:
        super().__init__()
        hidden_ch = int(hidden_ch)
        num_blocks = int(num_blocks)
        max_dilation = max(int(max_dilation), 1)
        if hidden_ch < 4:
            raise ValueError("hidden_ch must be >= 4")
        if num_blocks < 1:
            raise ValueError("num_blocks must be >= 1")

        self.in_proj = nn.Conv2d(4, hidden_ch, kernel_size=3, padding=1)
        dilations = []
        d = 1
        for _ in range(num_blocks):
            dilations.append(min(d, max_dilation))
            d *= 2
        self.blocks = nn.ModuleList(
            [_DilatedResidualBlock(hidden_ch, dilation=v) for v in dilations]
        )
        self.out_norm = nn.GroupNorm(1, hidden_ch)
        self.out_proj = nn.Conv2d(hidden_ch, 2, kernel_size=3, padding=1)
        nn.init.zeros_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)
        self.last_gate = None

    def forward(self, y_pre: torch.Tensor, y_noisy: torch.Tensor) -> torch.Tensor:
        feat = self._build_features(y_pre, y_noisy)
        h = self.in_proj(feat)
        for block in self.blocks:
            h = block(h)
        out = self.out_proj(F.silu(self.out_norm(h)))
        self.last_gate = None
        return self._to_complex(out, y_pre)


class ConfidenceGatedResidualPriorPredictor(nn.Module, _ResidualFeatureMixin):
    """Predict a local confidence map and gate the observable residual.

    The network predicts one real gate per TF position:
        G = sigmoid(H(Y_pre, R_obs)),  G in [0,1]
        R_prior = G * R_obs

    ``initial_gate`` controls the starting prior.  A small non-zero value is
    safer than 0.5 when R_obs is very noisy, while still providing useful
    gradients through the sigmoid head.
    """

    def __init__(
        self,
        hidden_ch: int = 48,
        num_blocks: int = 4,
        max_dilation: int = 8,
        initial_gate: float = 0.10,
    ) -> None:
        super().__init__()
        hidden_ch = int(hidden_ch)
        num_blocks = int(num_blocks)
        max_dilation = max(int(max_dilation), 1)
        initial_gate = float(initial_gate)
        if hidden_ch < 4:
            raise ValueError("hidden_ch must be >= 4")
        if num_blocks < 1:
            raise ValueError("num_blocks must be >= 1")
        if not 0.0 < initial_gate < 1.0:
            raise ValueError("initial_gate must be strictly between 0 and 1")

        self.in_proj = nn.Conv2d(4, hidden_ch, kernel_size=3, padding=1)
        dilations = []
        d = 1
        for _ in range(num_blocks):
            dilations.append(min(d, max_dilation))
            d *= 2
        self.blocks = nn.ModuleList(
            [_DilatedResidualBlock(hidden_ch, dilation=v) for v in dilations]
        )
        self.out_norm = nn.GroupNorm(1, hidden_ch)
        self.gate_proj = nn.Conv2d(hidden_ch, 1, kernel_size=3, padding=1)

        # Start from a spatially constant confidence prior rather than random
        # TF gates. This makes the first residual center deterministic/stable.
        nn.init.zeros_(self.gate_proj.weight)
        gate_bias = math.log(initial_gate / (1.0 - initial_gate))
        nn.init.constant_(self.gate_proj.bias, gate_bias)
        self.last_gate = None

    def forward(self, y_pre: torch.Tensor, y_noisy: torch.Tensor) -> torch.Tensor:
        if y_pre.shape != y_noisy.shape:
            raise ValueError(
                f"y_pre and y_noisy must have identical shapes, got "
                f"{tuple(y_pre.shape)} and {tuple(y_noisy.shape)}"
            )
        r_obs = y_noisy - y_pre
        feat = self._build_features(y_pre, y_noisy)
        h = self.in_proj(feat)
        for block in self.blocks:
            h = block(h)
        logits = self.gate_proj(F.silu(self.out_norm(h)))
        gate = torch.sigmoid(logits)
        self.last_gate = gate
        return r_obs * gate.to(dtype=r_obs.real.dtype if torch.is_complex(r_obs) else r_obs.dtype)

    @staticmethod
    def oracle_gate(
        r_target: torch.Tensor,
        r_obs: torch.Tensor,
        patch_freq: int = 5,
        patch_time: int = 3,
        floor_ratio: float = 1e-4,
        eps: float = 1e-8,
    ) -> torch.Tensor:
        """Least-squares real gate for R_prior = G * R_obs.

        For each local TF neighborhood, the unconstrained optimal real gain is
            Re(conj(R_obs) * R_target) / (|R_obs|^2 + eps).
        It is clipped to [0,1] to retain the interpretation as confidence.

        Local pooling makes the oracle less sensitive to single-bin phase/STFT
        noise.  Near-silent neighborhoods are assigned zero confidence.
        """
        if r_target.shape != r_obs.shape:
            raise ValueError("r_target and r_obs must have identical shapes")
        if r_target.dim() != 4:
            raise ValueError(f"Expected [B,C,F,T], got {tuple(r_target.shape)}")

        pf = max(1, int(patch_freq))
        pt = max(1, int(patch_time))
        if pf % 2 == 0 or pt % 2 == 0:
            raise ValueError("oracle gate patch sizes must be odd for shape-preserving pooling")

        if torch.is_complex(r_obs):
            cross = (torch.conj(r_obs) * r_target).real
        else:
            cross = r_obs * r_target
        obs_power = torch.abs(r_obs).pow(2)

        # One shared confidence map across spectral channels.
        cross = cross.float().sum(dim=1, keepdim=True)
        obs_power = obs_power.float().sum(dim=1, keepdim=True)

        if pf > 1 or pt > 1:
            pad = (pt // 2, pt // 2, pf // 2, pf // 2)
            # Replication avoids artificial low-energy borders from zero padding.
            cross = F.avg_pool2d(F.pad(cross, pad, mode="replicate"), (pf, pt), stride=1)
            obs_power = F.avg_pool2d(
                F.pad(obs_power, pad, mode="replicate"), (pf, pt), stride=1
            )

        sample_power = obs_power.mean(dim=(2, 3), keepdim=True)
        power_floor = (float(floor_ratio) * sample_power).clamp_min(float(eps))
        valid = obs_power > power_floor
        gate = cross / obs_power.clamp_min(float(eps))
        gate = gate.clamp(0.0, 1.0)
        return torch.where(valid, gate, torch.zeros_like(gate))


def build_residual_prior_predictor(
    prior_type: str,
    hidden_ch: int = 48,
    num_blocks: int = 4,
    max_dilation: int = 8,
    initial_gate: float = 0.10,
) -> nn.Module:
    mode = str(prior_type).strip().lower()
    if mode == "direct":
        return DirectResidualInitPredictor(
            hidden_ch=hidden_ch,
            num_blocks=num_blocks,
            max_dilation=max_dilation,
        )
    if mode == "confidence_gate":
        return ConfidenceGatedResidualPriorPredictor(
            hidden_ch=hidden_ch,
            num_blocks=num_blocks,
            max_dilation=max_dilation,
            initial_gate=initial_gate,
        )
    raise ValueError(f"Unknown residual prior type: {prior_type!r}")


# Backward-compatible import used by older checkpoints/source files.
ResidualInitPredictor = DirectResidualInitPredictor
