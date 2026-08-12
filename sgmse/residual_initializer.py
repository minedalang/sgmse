"""Deterministic residual initializer for predictive-residual diffusion refinement.

The module predicts the clean correction residual
    R* = X_clean - Y_pre
from quantities available at inference time: the frontend estimate Y_pre and
original noisy spectrum Y_noisy.  The predicted residual is used as the mean
(anchor) of an OUVE residual diffusion process; it is not the final output by
itself.
"""

from __future__ import annotations

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


class ResidualInitPredictor(nn.Module):
    """Predict a complex correction residual from frontend/noisy spectra.

    Input features are
        Re(Y_pre), Im(Y_pre), Re(Y_noisy-Y_pre), Im(Y_noisy-Y_pre).
    The final convolution is zero-initialized so enabling the module starts from
    a conservative zero residual and learns the initializer from supervision.
    """

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

        self.hidden_ch = hidden_ch
        self.num_blocks = num_blocks
        self.max_dilation = max_dilation

        self.in_proj = nn.Conv2d(4, hidden_ch, kernel_size=3, padding=1)
        dilations = []
        d = 1
        for _ in range(num_blocks):
            dilations.append(min(d, max_dilation))
            d *= 2
        self.blocks = nn.ModuleList(
            [_DilatedResidualBlock(hidden_ch, dilation=dilation) for dilation in dilations]
        )
        self.out_norm = nn.GroupNorm(1, hidden_ch)
        self.out_proj = nn.Conv2d(hidden_ch, 2, kernel_size=3, padding=1)
        nn.init.zeros_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)

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

    def forward(self, y_pre: torch.Tensor, y_noisy: torch.Tensor) -> torch.Tensor:
        if y_pre.shape != y_noisy.shape:
            raise ValueError(
                f"y_pre and y_noisy must have identical shapes, got "
                f"{tuple(y_pre.shape)} and {tuple(y_noisy.shape)}"
            )
        residual_obs = y_noisy - y_pre
        feat = torch.cat([self._to_ri(y_pre), self._to_ri(residual_obs)], dim=1).float()
        h = self.in_proj(feat)
        for block in self.blocks:
            h = block(h)
        out = self.out_proj(F.silu(self.out_norm(h)))
        return self._to_complex(out, y_pre)
