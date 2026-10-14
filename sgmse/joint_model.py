"""sgmse.joint_model

联合训练 LightningModule：Frontend(前端) + ScoreModel/SGMSE(后端)。

核心思想：
- batch 提供 clean/noisy 波形
- noisy 波形先过 frontend 得到 y_pre (可微)
- SGMSE 的 score network 接收 (y=noisy, y_cond=y_pre) 或 (y=y_pre, y_cond=noisy)
- 使用 SGMSE 标准训练目标（score-matching / denoiser loss）
- loss 反向传播同时更新 SEMamba 与 SGMSE

实现细节：
- 由于 SEMamba 仓库的 generator.py 使用相对导入（.mamba_block 等），
  这里用“虚拟包”方式从 semamba_root/models 动态导入，避免强制改目录名。
"""

from __future__ import annotations
import warnings
import torchaudio.functional as AF
from transformers import WavLMModel
import argparse
import importlib
import sys
import types
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
import torch.distributed as dist
from torchaudio import load,save
from librosa import resample
from pesq import pesq
from pystoi import stoi
from sgmse.util.other import si_sdr, pad_spec

from sgmse.model import ScoreModel
from sgmse.sdes import SBVESDE
from sgmse.frontends import build_frontend
from sgmse.anchor_v2 import BandTimeAnchorV2Controller, solve_two_direction_tile_oracle
from sgmse.reliability_router import BandTimeReliabilityRouter, solve_frontend_projection_oracle
from sgmse.pathsim_vA import BandTimeVariancePathEstimator, load_pathsim_va_checkpoint
from sgmse.residual_initializer import (
    ConfidenceGatedResidualPriorPredictor,
    build_residual_prior_predictor,
)

from contextlib import contextmanager
from functools import partial
import copy
import os
import time

def _load_yaml(path: str) -> Dict[str, Any]:
    try:
        import yaml
    except Exception as e:  # pragma: no cover
        raise ImportError("PyYAML is required for loading SEMamba config.yaml") from e
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _import_semamba_package(models_dir: Path, package_name: str = "_semamba_pkg"):
    """把 models_dir 注册成一个可 import 的“虚拟包”，用于支持 generator.py 的相对导入。"""
    if package_name not in sys.modules:
        pkg = types.ModuleType(package_name)
        pkg.__path__ = [str(models_dir)]  # make it a package
        sys.modules[package_name] = pkg
    # 强制让 import machinery 看到这个路径
    if str(models_dir) not in sys.path:
        sys.path.insert(0, str(models_dir))
    # 返回包名
    return package_name


def _load_semamba(semamba_root: str, config_path: str, ckpt_path: Optional[str]) -> Tuple[torch.nn.Module, Dict[str, Any], Any, Any]:
    """返回: (semamba_model, cfg_dict, mag_phase_stft_fn, mag_phase_istft_fn)"""
    semamba_root = str(semamba_root)
    cfg = _load_yaml(config_path)

    models_dir = Path(semamba_root) / "models"
    if not models_dir.exists():
        raise FileNotFoundError(f"SEMamba models dir not found: {models_dir}")

    pkg = _import_semamba_package(models_dir)

    gen = importlib.import_module(f"{pkg}.generator")
    stfts = importlib.import_module(f"{pkg}.stfts")

    if not hasattr(gen, "SEMamba"):
        raise RuntimeError("Cannot find SEMamba class in generator.py")
    if not hasattr(stfts, "mag_phase_stft") or not hasattr(stfts, "mag_phase_istft"):
        raise RuntimeError("Cannot find mag_phase_stft/mag_phase_istft in stfts.py")

    model = gen.SEMamba(cfg)

    if ckpt_path:
        ckpt = torch.load(ckpt_path, map_location="cpu")
        # 常见保存形式：直接 state_dict，或者 {"generator": state_dict}
        if isinstance(ckpt, dict) and "state_dict" in ckpt:
            state = ckpt["state_dict"]
        elif isinstance(ckpt, dict) and "generator" in ckpt:
            state = ckpt["generator"]
        elif isinstance(ckpt, dict) and "model" in ckpt:
            state = ckpt["model"]
        else:
            state = ckpt
        missing, unexpected = model.load_state_dict(state, strict=False)
        if missing or unexpected:
            print(f"[SEMamba] load_state_dict strict=False: missing={len(missing)} unexpected={len(unexpected)}")

    return model, cfg, stfts.mag_phase_stft, stfts.mag_phase_istft


def _match_length(y: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
    """把 y 的最后一维长度调成 ref 的长度（center crop/pad），y/ref: [B, 1, T]."""
    T = ref.size(-1)
    cur = y.size(-1)
    if cur == T:
        return y
    if cur > T:
        start = (cur - T) // 2
        return y[..., start : start + T]
    pad = T - cur
    left = pad // 2
    right = left + (pad % 2)
    return F.pad(y, (left, right), mode="constant", value=0.0)





class StateAwareAnchorController(nn.Module):
    """Global state-aware controller for weak-anchor OUVE mean calibration.

    It predicts two utterance-level scalars from the current sampling state and
    conditional spectra:
        c_t: anchor composition coefficient, A_t = c_t Y_pre + (1-c_t) Y_noisy
        g_t: correction reliability, K_t = K_max * E(t) * g_t

    This is the original/global controller.  It is kept for checkpoint
    compatibility and for ablation against the new band-wise controller.
    """

    controller_type = "global_mlp"

    def __init__(
        self,
        hidden_ch: int = 64,
        k_max: float = 0.08,
        time_power: float = 1.0,
        infer_k_scale: float = 1.0,
    ):
        super().__init__()
        self.hidden_ch = int(hidden_ch)
        self.k_max = float(k_max)
        self.time_power = float(time_power)
        self.infer_k_scale = float(infer_k_scale)
        # Four maps (d_pre, d_noisy, residual, x_to_base), each with mean/std/max, plus t_norm.
        in_dim = 4 * 3 + 1
        self.net = nn.Sequential(
            nn.Linear(in_dim, self.hidden_ch),
            nn.SiLU(),
            nn.Linear(self.hidden_ch, self.hidden_ch),
            nn.SiLU(),
            nn.Linear(self.hidden_ch, 2),
        )

    @staticmethod
    def _abs_map(x: torch.Tensor) -> torch.Tensor:
        return torch.abs(x) if torch.is_complex(x) else x.abs()

    @classmethod
    def _stats(cls, x: torch.Tensor) -> torch.Tensor:
        x = cls._abs_map(x).float().flatten(start_dim=1)
        return torch.stack([
            x.mean(dim=1),
            x.std(dim=1, unbiased=False),
            x.amax(dim=1),
        ], dim=1)

    def _time_envelope(self, t: torch.Tensor, t_max: float = 1.0) -> torch.Tensor:
        t_norm = (t.float() / max(float(t_max), 1e-8)).clamp(0.0, 1.0)
        s = t_norm.pow(float(self.time_power))
        return (4.0 * s * (1.0 - s)).clamp(0.0, 1.0)

    def forward(
        self,
        x_t: torch.Tensor,
        mu_base: torch.Tensor,
        y_pre: torch.Tensor,
        y_noisy: torch.Tensor,
        t: torch.Tensor,
        t_max: float = 1.0,
        detach_inputs: bool = True,
        k_max_override: Optional[float] = None,
        k_scale: Optional[float] = None,
    ) -> Dict[str, torch.Tensor]:
        if detach_inputs:
            x_t = x_t.detach()
            mu_base = mu_base.detach()
            y_pre = y_pre.detach()
            y_noisy = y_noisy.detach()
            t = t.detach()

        if t.dim() == 0:
            t = torch.ones(x_t.shape[0], device=x_t.device, dtype=torch.float32) * t
        t = t.to(device=x_t.device, dtype=torch.float32).flatten()

        d_pre = y_pre - mu_base
        d_noisy = y_noisy - mu_base
        residual = y_noisy - y_pre
        x_to_base = x_t - mu_base

        feat = torch.cat([
            self._stats(d_pre),
            self._stats(d_noisy),
            self._stats(residual),
            self._stats(x_to_base),
            (t / max(float(t_max), 1e-8)).clamp(0.0, 1.0)[:, None],
        ], dim=1).to(dtype=torch.float32)

        raw = self.net(feat)
        out_dtype = mu_base.real.dtype if torch.is_complex(mu_base) else mu_base.dtype
        c_t = torch.sigmoid(raw[:, 0]).view(-1, 1, 1, 1).to(dtype=out_dtype)
        g_t = torch.sigmoid(raw[:, 1]).view(-1, 1, 1, 1).to(dtype=out_dtype)
        E_t = self._time_envelope(t, t_max=t_max).view(-1, 1, 1, 1).to(device=x_t.device, dtype=g_t.dtype)
        k_max_eff = float(self.k_max if k_max_override is None else k_max_override)
        scale_eff = float(self.infer_k_scale if k_scale is None else k_scale)
        k_cap = max(k_max_eff * scale_eff, 0.0)
        K_t = (k_cap * E_t * g_t).clamp(0.0, k_cap)

        A_t = c_t * y_pre + (1.0 - c_t) * y_noisy
        delta_mu = K_t * (A_t - mu_base)
        mu_anchor = mu_base + delta_mu
        return {
            "mu_anchor": mu_anchor,
            "delta_mu": delta_mu,
            "A_t": A_t,
            "K_t": K_t,
            "c_t": c_t,
            "g_t": g_t,
            "E_t": E_t,
        }


class StateAwareBandAnchorController(nn.Module):
    """Lightweight band-wise state-aware controller for OUVE mean calibration.

    It keeps the sampling-side formulation unchanged, but predicts band-wise
    coefficients instead of one utterance-level scalar:
        c_t(f): band-wise anchor composition
        g_t(f): band-wise correction reliability

    The network is still tiny.  For each band it computes statistics of four
    deviation maps and applies the same shared MLP to every band.
    """

    controller_type = "band_mlp"

    def __init__(
        self,
        hidden_ch: int = 32,
        k_max: float = 0.08,
        time_power: float = 1.0,
        num_bands: int = 32,
        infer_k_scale: float = 1.0,
    ):
        super().__init__()
        self.hidden_ch = int(hidden_ch)
        self.k_max = float(k_max)
        self.time_power = float(time_power)
        self.num_bands = max(int(num_bands), 1)
        self.infer_k_scale = float(infer_k_scale)
        in_dim = 4 * 3 + 1
        self.net = nn.Sequential(
            nn.Linear(in_dim, self.hidden_ch),
            nn.SiLU(),
            nn.Linear(self.hidden_ch, self.hidden_ch),
            nn.SiLU(),
            nn.Linear(self.hidden_ch, 2),
        )

    @staticmethod
    def _abs_map(x: torch.Tensor) -> torch.Tensor:
        return torch.abs(x) if torch.is_complex(x) else x.abs()

    @staticmethod
    def _band_edges(freq_bins: int, num_bands: int, device=None) -> torch.Tensor:
        nb = max(1, min(int(num_bands), int(freq_bins)))
        return torch.linspace(0, int(freq_bins), nb + 1, device=device).round().long()

    @classmethod
    def _band_stats(cls, x: torch.Tensor, num_bands: int) -> torch.Tensor:
        """Return [B, num_bands_eff, 3] with mean/std/max per frequency band."""
        mag = cls._abs_map(x).float()
        B = mag.shape[0]
        freq_bins = int(mag.shape[-2])
        edges = cls._band_edges(freq_bins, num_bands, device=mag.device)
        stats = []
        for i in range(edges.numel() - 1):
            lo = int(edges[i].item())
            hi = int(edges[i + 1].item())
            if hi <= lo:
                hi = min(lo + 1, freq_bins)
            band = mag[..., lo:hi, :].reshape(B, -1)
            stats.append(torch.stack([
                band.mean(dim=1),
                band.std(dim=1, unbiased=False),
                band.amax(dim=1),
            ], dim=1))
        return torch.stack(stats, dim=1)

    @classmethod
    def _expand_bands(cls, x_band: torch.Tensor, freq_bins: int) -> torch.Tensor:
        """Nearest-expand [B,1,N_band,1] to [B,1,F,1]."""
        if x_band.shape[-2] == int(freq_bins):
            return x_band
        return F.interpolate(x_band, size=(int(freq_bins), 1), mode="nearest")

    def _time_envelope(self, t: torch.Tensor, t_max: float = 1.0) -> torch.Tensor:
        t_norm = (t.float() / max(float(t_max), 1e-8)).clamp(0.0, 1.0)
        s = t_norm.pow(float(self.time_power))
        return (4.0 * s * (1.0 - s)).clamp(0.0, 1.0)

    def forward(
        self,
        x_t: torch.Tensor,
        mu_base: torch.Tensor,
        y_pre: torch.Tensor,
        y_noisy: torch.Tensor,
        t: torch.Tensor,
        t_max: float = 1.0,
        detach_inputs: bool = True,
        k_max_override: Optional[float] = None,
        k_scale: Optional[float] = None,
    ) -> Dict[str, torch.Tensor]:
        if detach_inputs:
            x_t = x_t.detach()
            mu_base = mu_base.detach()
            y_pre = y_pre.detach()
            y_noisy = y_noisy.detach()
            t = t.detach()

        if t.dim() == 0:
            t = torch.ones(x_t.shape[0], device=x_t.device, dtype=torch.float32) * t
        t = t.to(device=x_t.device, dtype=torch.float32).flatten()

        d_pre = y_pre - mu_base
        d_noisy = y_noisy - mu_base
        residual = y_noisy - y_pre
        x_to_base = x_t - mu_base
        freq_bins = int(mu_base.shape[-2])
        nb = max(1, min(int(self.num_bands), freq_bins))

        # [B, N_band, 12]
        feat_maps = torch.cat([
            self._band_stats(d_pre, nb),
            self._band_stats(d_noisy, nb),
            self._band_stats(residual, nb),
            self._band_stats(x_to_base, nb),
        ], dim=-1)
        t_feat = (t / max(float(t_max), 1e-8)).clamp(0.0, 1.0).view(-1, 1, 1).expand(-1, nb, 1)
        feat = torch.cat([feat_maps, t_feat], dim=-1).to(dtype=torch.float32)

        raw = self.net(feat.reshape(-1, feat.shape[-1])).view(x_t.shape[0], nb, 2)
        out_dtype = mu_base.real.dtype if torch.is_complex(mu_base) else mu_base.dtype
        c_band = torch.sigmoid(raw[..., 0]).view(-1, 1, nb, 1).to(dtype=out_dtype)
        g_band = torch.sigmoid(raw[..., 1]).view(-1, 1, nb, 1).to(dtype=out_dtype)
        c_t = self._expand_bands(c_band, freq_bins=freq_bins)
        g_t = self._expand_bands(g_band, freq_bins=freq_bins)

        E_t = self._time_envelope(t, t_max=t_max).view(-1, 1, 1, 1).to(device=x_t.device, dtype=g_t.dtype)
        k_max_eff = float(self.k_max if k_max_override is None else k_max_override)
        scale_eff = float(self.infer_k_scale if k_scale is None else k_scale)
        k_cap = max(k_max_eff * scale_eff, 0.0)
        K_band = (k_cap * E_t * g_band).clamp(0.0, k_cap)
        K_t = self._expand_bands(K_band, freq_bins=freq_bins)

        A_t = c_t * y_pre + (1.0 - c_t) * y_noisy
        delta_mu = K_t * (A_t - mu_base)
        mu_anchor = mu_base + delta_mu
        return {
            "mu_anchor": mu_anchor,
            "delta_mu": delta_mu,
            "A_t": A_t,
            "K_t": K_t,
            "c_t": c_t,
            "g_t": g_t,
            "E_t": E_t,
            "c_band": c_band,
            "g_band": g_band,
            "K_band": K_band,
        }


class JointFrontendSgmseModel(ScoreModel):
    strict_loading = False
    @staticmethod
    def add_argparse_args(parser):
        # 继承 ScoreModel 的参数
        ScoreModel.add_argparse_args(parser)

        # 可插拔前端相关参数
        parser.add_argument("--frontend", type=str, choices=("semamba", "storm", "tfgridnet"), default="semamba")
        parser.add_argument("--frontend_root", type=str, default=None, help="Frontend repo root (Semamba/Storm/TF-GridNet)")
        parser.add_argument("--frontend_config", type=str, default=None, help="Frontend config path (.yaml/.yml/.json)")
        parser.add_argument("--frontend_ckpt", type=str, default=None, help="Optional frontend checkpoint")
        parser.add_argument("--frontend_module", type=str, default=None, help="For storm/tfgridnet: python module name to import")
        parser.add_argument("--frontend_class", type=str, default=None, help="For storm/tfgridnet: class/builder name inside frontend_module")
        parser.add_argument("--frontend_state_dict_key", type=str, default=None, help="Optional checkpoint key for external frontends")
        parser.add_argument("--frontend_trainable", action="store_true", help="Enable joint training for the frontend")
        parser.add_argument(
            "--frontend_aux_loss",
            type=str,
            choices=("none", "time_l1","time_mse", "spec_mse", "spec_mae", "spec_cplx_mag"),
            default="none",
            help="Denoiser loss type used in joint training",
        )
        parser.add_argument(
            "--weighting_denoiser_to_score",
            type=float,
            default=0.5,
            help="Total loss = a * denoiser_loss + (1-a) * score_loss",
        )

        # ===== Frontend-relative no-regret / regret-aware score reweighting =====
        parser.add_argument(
            "--frontend_nr_strategy",
            type=str,
            choices=("auxiliary", "score_reweight", "adaptive_score_guidance"),
            default="auxiliary",
            help=(
                "How local frontend-relative regret affects optimization. "
                "'auxiliary' keeps the legacy extra NR loss; 'score_reweight' uses the "
                "detached local degradation map to reweight the main score-matching TF loss; "
                "'adaptive_score_guidance' keeps DSM unchanged and adds a reliability-gated "
                "frontend-derived target directly in score space."
            ),
        )
        parser.add_argument(
            "--frontend_nr_reweight_strength",
            type=float,
            default=1.0,
            help=(
                "Average extra score-loss weight assigned to detected hard TF patches in "
                "score_reweight mode. 1.0 makes a typical violated patch about 2x weighted."
            ),
        )
        parser.add_argument(
            "--frontend_nr_reweight_max",
            type=float,
            default=4.0,
            help="Maximum per-TF score-loss multiplier in score_reweight mode.",
        )
        parser.add_argument(
            "--frontend_nr_guidance_strength",
            type=float,
            default=0.25,
            help=(
                "Lambda for adaptive_score_guidance. The added score-space alignment loss is "
                "lambda * gate * ||sigma*(s_theta-s_front)||^2 / 2. A conservative first sweep "
                "is 0.1/0.25/0.5."
            ),
        )
        parser.add_argument(
            "--frontend_nr_guidance_temperature",
            type=float,
            default=0.25,
            help=(
                "Temperature of the smooth reliability gate in log error-ratio space. Smaller "
                "values make the fallback toward the frontend more switch-like."
            ),
        )
        parser.add_argument(
            "--frontend_nr_guidance_tail_fraction",
            type=float,
            default=1.0,
            help=(
                "Fraction of locally degraded TF patches eligible for adaptive score guidance. "
                "1.0 uses all violated patches; values <1 keep only the worst tail for ablation."
            ),
        )
        parser.add_argument(
            "--frontend_nr_guidance_coverage_target",
            type=float,
            default=0.0,
            help=(
                "Target effective TF coverage for adaptive score guidance. 0 disables coverage "
                "calibration and reproduces the original summed guidance loss. Values such as "
                "0.05 compensate for sparse gate activation; the scale is capped internally."
            ),
        )

        # ===== RRSD: Oracle-distilled reliability-routed diffusion =====
        parser.add_argument("--rrsd_weight", type=float, default=0.0,
                            help="Training weight for the RRSD reliability-router distillation loss. 0 disables RRSD training.")
        parser.add_argument("--rrsd_hidden_ch", type=int, default=48)
        parser.add_argument("--rrsd_num_bands", type=int, default=32)
        parser.add_argument("--rrsd_num_time_blocks", type=int, default=16)
        parser.add_argument("--rrsd_init_alpha", type=float, default=0.05,
                            help="Initial routing coefficient. Small values make a newly-added router close to a no-op.")
        parser.add_argument("--rrsd_benefit_weight", type=float, default=4.0,
                            help="Extra oracle-distillation weight on tiles with large clean-referenced routing benefit.")
        parser.add_argument("--rrsd_recon_weight", type=float, default=0.10,
                            help="Weight of normalized x0 reconstruction loss for routed estimates.")
        parser.add_argument("--rrsd_smooth_weight", type=float, default=0.01,
                            help="Spatial smoothness weight on the low-resolution alpha routing grid.")
        parser.add_argument("--rrsd_oracle_eps", type=float, default=1e-6)
        parser.add_argument(
            "--rrsd_teacher", type=str, default="projection", choices=["projection", "kstep", "future_grad"],
            help=(
                "RRSD training objective. 'projection' uses the original x0 projection oracle; "
                "'kstep' uses a discrete counterfactual lookahead teacher; "
                "'future_grad' lets the router emit one routing action and backpropagates a one-step "
                "future re-score loss through the frozen diffusion network input."
            ),
        )
        parser.add_argument("--rrsd_lookahead_k", type=int, default=1,
                            help="Number of deterministic reverse-mean transitions used by the K-step teacher.")
        parser.add_argument(
            "--rrsd_lookahead_candidates", type=str, default="0,0.25,0.5,0.75,1.0",
            help="Comma-separated global scales applied to the local projection alpha map by the K-step teacher.",
        )
        parser.add_argument(
            "--rrsd_lookahead_dt", type=float, default=0.0,
            help="Lookahead reverse-step size. <=0 uses (T-t_eps)/(sde.N-1), matching the ordinary PC grid spacing.",
        )
        parser.add_argument(
            "--rrsd_lookahead_chunk", type=int, default=0,
            help=(
                "Candidate chunk size for the vectorized K-step teacher. 0 batches all candidates "
                "into one enlarged batch; use a positive value if GPU memory is insufficient."
            ),
        )
        parser.add_argument("--rrsd_future_weight", type=float, default=1.0,
                            help="Weight of the one-step future re-score loss in --rrsd_teacher future_grad mode.")
        parser.add_argument("--rrsd_future_gate_cost", type=float, default=1e-3,
                            help="Small mean-gate intervention cost used only by future_grad + gate.")
        parser.add_argument(
            "--rrsd_use_gate", action="store_true",
            help=(
                "Use a shared-backbone gate+strength router: alpha_eff = g * a. "
                "With projection/kstep, g has an oracle-derived target; with future_grad, g and strength "
                "are optimized jointly through the future loss and a small gate intervention cost."
            ),
        )
        parser.add_argument("--rrsd_init_gate", type=float, default=0.20,
                            help="Initial sigmoid probability of the explicit gate when --rrsd_use_gate is enabled.")
        parser.add_argument("--rrsd_gate_threshold", type=float, default=0.01,
                            help="Minimum K-step relative trajectory gain required for a positive gate label; used only with --rrsd_teacher kstep.")
        parser.add_argument("--rrsd_gate_local_threshold", type=float, default=0.005,
                            help="Minimum local projection benefit ratio required for a positive band-time gate label. This is the gate-only threshold with --rrsd_teacher projection.")
        parser.add_argument("--rrsd_gate_weight", type=float, default=1.0,
                            help="Weight of gate BCE when --rrsd_use_gate is enabled.")
        parser.add_argument("--rrsd_strength_weight", type=float, default=1.0,
                            help="Weight of conditional strength distillation when --rrsd_use_gate is enabled.")
        parser.add_argument("--rrsd_gate_pos_weight", type=float, default=2.0,
                            help="Extra BCE weight for positive gate tiles.")
        parser.add_argument("--rrsd_validate", action="store_true",
                            help="Use RRSD during the periodic file-level validation sampler.")
        parser.add_argument("--rrsd_freeze_base", action="store_true",
                            help="Freeze the frontend and score network; train only the RRSD router. Recommended for projection-gate and required for K-step teacher experiments.")
        parser.add_argument(
            "--frontend_nr_weight",
            type=float,
            default=0.0,
            help=(
                "Weight of the LEGACY auxiliary frontend-relative no-regret loss. Used only "
                "when --frontend_nr_strategy auxiliary. Set 0 in score_reweight mode."
            ),
        )
        parser.add_argument(
            "--frontend_nr_margin",
            type=float,
            default=0.05,
            help=(
                "Relative error margin delta. A penalty is applied only when "
                "E_diff > (1+delta)*E_front."
            ),
        )
        parser.add_argument(
            "--frontend_nr_t_max",
            type=float,
            default=0.30,
            help=(
                "Apply the no-regret loss only for diffusion times t <= this value. "
                "For the default OUVE T=1, 0.30 protects the low-noise 30%% interval."
            ),
        )
        parser.add_argument(
            "--frontend_nr_eps",
            type=float,
            default=1e-8,
            help="Numerical epsilon used by frontend-relative no-regret diagnostics.",
        )
        parser.add_argument(
            "--frontend_nr_mode",
            type=str,
            choices=("global", "patch"),
            default="global",
            help=(
                "No-regret granularity. 'global' reproduces the original utterance/sample-level "
                "constraint; 'patch' uses local time-frequency relative degradation with hard-region mining."
            ),
        )
        parser.add_argument(
            "--frontend_nr_patch_freq",
            type=int,
            default=5,
            help="Odd frequency-axis pooling size for patch no-regret (e.g. 5).",
        )
        parser.add_argument(
            "--frontend_nr_patch_time",
            type=int,
            default=3,
            help="Odd time-axis pooling size for patch no-regret (e.g. 3).",
        )
        parser.add_argument(
            "--frontend_nr_topk",
            type=float,
            default=0.20,
            help=(
                "Fraction of the most severely degraded local patches kept by patch no-regret. "
                "Must be in (0, 1]; 1 keeps all patches."
            ),
        )
        parser.add_argument(
            "--frontend_nr_floor_ratio",
            type=float,
            default=0.01,
            help=(
                "Floor used only when computing the local backend/frontend error ratio: "
                "denom=max(E_front_patch, floor_ratio*mean(E_front_patch)). "
                "This prevents near-silent bins from producing huge ratios."
            ),
        )

        # 配置用于内容对齐loss的预训练模型
        parser.add_argument("--content_model_name", type=str, default="microsoft/wavlm-base-plus",
                            help="Frozen SSL model for content loss.")
        parser.add_argument("--content_layer", type=int, default=9,
                            help="Which hidden layer of WavLM to use for content loss. Middle layers usually work best.")
        parser.add_argument("--content_loss_type", type=str, default="smooth_l1",
                            choices=("l1", "mse", "smooth_l1", "cosine"),
                            help="Regression loss on SSL hidden states.")
        parser.add_argument("--final_content_weight", type=float, default=0.0,
                            help="Weight of content loss on final enhanced result.")
        parser.add_argument("--selector_content_weight", type=float, default=0.0,
                            help="Weight of content loss on selector y_hint.")

        # ===== TF-grid selector oracle mask loss =====
        parser.add_argument("--selector_oracle_weight", type=float, default=0.0,
                            help="Weight of projection-oracle mask loss for tfgrid selector. 0 disables it.")
        parser.add_argument("--selector_oracle_conf_weight", type=float, default=1.0,
                            help="0 = unweighted mask loss; 1 = fully confidence-weighted mask loss.")
        parser.add_argument("--selector_oracle_eps", type=float, default=1e-8,
                            help="Numerical epsilon used in selector projection oracle.")
        
        
        # ===== State-aware conditional anchor calibration (sampler-side) =====
        parser.add_argument("--anchor_calib_weight", type=float, default=0.0,
                            help="Weight of state-aware anchor calibration loss. 0 disables the controller.")
        parser.add_argument("--anchor_controller_type", type=str, default="global_mlp",
                            choices=("global_mlp", "band_mlp"),
                            help="Type of anchor controller: global scalar MLP or lightweight band-wise shared MLP.")
        parser.add_argument("--anchor_controller_hidden_ch", type=int, default=64,
                            help="Hidden channels of the state-aware anchor controller. For band_mlp, 32 is usually enough.")
        parser.add_argument("--anchor_num_bands", type=int, default=32,
                            help="Number of frequency bands for --anchor_controller_type band_mlp.")
        parser.add_argument("--anchor_k_max", type=float, default=0.08,
                            help="Maximum pull strength K_max used by sampler mean calibration.")
        parser.add_argument("--anchor_time_power", type=float, default=1.0,
                            help="Power p for the time envelope E(t)=4s(1-s), s=(t/T)^p.")
        parser.add_argument("--anchor_band_smooth_weight", type=float, default=0.0,
                            help="Optional smoothness penalty for neighboring band-wise c/g outputs. Only used by band_mlp.")
        parser.add_argument("--anchor_loss_weight_c", type=float, default=1.0,
                            help="Weight of oracle anchor-composition loss for c_t.")
        parser.add_argument("--anchor_loss_weight_k", type=float, default=1.0,
                            help="Weight of oracle pull-strength loss for K_t.")
        parser.add_argument("--anchor_loss_weight_recon", type=float, default=0.1,
                            help="Weight of direct mean calibration loss ||mu_anchor-X||.")

        # ===== Anchor V2: fine-grained band-time x0 residual calibration =====
        parser.add_argument("--anchor_v2_weight", type=float, default=0.0,
                            help="Weight of Anchor V2 loss. 0 disables the independent V2 controller.")
        parser.add_argument("--anchor_v2_hidden_ch", type=int, default=48,
                            help="Hidden channels of the band-time Anchor V2 controller.")
        parser.add_argument("--anchor_v2_num_bands", type=int, default=32,
                            help="Number of adaptive frequency bands used by Anchor V2.")
        parser.add_argument("--anchor_v2_num_time_blocks", type=int, default=16,
                            help="Number of adaptive time blocks used by Anchor V2.")
        parser.add_argument("--anchor_v2_a_max", type=float, default=0.25,
                            help="Maximum signed coefficient for Y_pre-x0_base evidence.")
        parser.add_argument("--anchor_v2_b_max", type=float, default=0.15,
                            help="Maximum signed coefficient for Y_noisy-Y_pre evidence.")
        parser.add_argument("--anchor_v2_time_power", type=float, default=1.0,
                            help="Power p in the V2 time envelope E(t)=4s(1-s), s=(t/T)^p.")
        parser.add_argument("--anchor_v2_loss_weight_a", type=float, default=1.0,
                            help="Weight of the local oracle loss for the Y_pre gate a.")
        parser.add_argument("--anchor_v2_loss_weight_b", type=float, default=1.0,
                            help="Weight of the local oracle loss for the noisy-residual gate b.")
        parser.add_argument("--anchor_v2_loss_weight_recon", type=float, default=0.1,
                            help="Weight of clean x0 reconstruction loss after V2 calibration.")
        parser.add_argument("--anchor_v2_loss_weight_gain", type=float, default=0.1,
                            help="Penalty when V2 increases clean-reference x0 error over the base score estimate.")
        parser.add_argument("--anchor_v2_smooth_weight", type=float, default=0.01,
                            help="Total-variation penalty over neighboring band-time gates.")
        parser.add_argument("--anchor_v2_sparsity_weight", type=float, default=0.001,
                            help="Small L1 penalty encouraging minimal conditional correction.")
        parser.add_argument("--anchor_v2_oracle_eps", type=float, default=1e-6,
                            help="Ridge/epsilon for the local two-direction projection oracle.")
        parser.add_argument("--anchor_v2_validate", action="store_true",
                            help="Use Anchor V2 in the periodic file-level validation sampler.")
        parser.add_argument("--anchor_v2_single_shot_train", action="store_true",
                            help="Train Anchor V2 only near one reverse-process location instead of sampling t uniformly.")
        parser.add_argument("--anchor_v2_single_shot_ratio", type=float, default=0.5,
                            help="Single-shot reverse progress in [0,1]: 0 is the first/high-noise predictor step and 1 is the final/low-noise step.")
        parser.add_argument("--anchor_v2_single_shot_train_width", type=float, default=0.10,
                            help="Half-width around --anchor_v2_single_shot_ratio used to sample Anchor V2 training times.")
        parser.add_argument("--anchor_v2_validate_single_shot", action="store_true",
                            help="When validating Anchor V2, inject it at only the configured single-shot location.")

        # ===== Path Simulation Version A: learned band-time variance shift =====
        parser.add_argument("--pathsim_va_enable", action="store_true",
                            help="Train the score model on a mixture of analytical and PathSim V-A states.")
        parser.add_argument("--pathsim_va_ckpt", type=str, default=None,
                            help="Checkpoint of the independently trained variance path estimator.")
        parser.add_argument("--pathsim_va_prob", type=float, default=0.3,
                            help="Per-sample probability of replacing a standard marginal state by a simulated state.")
        parser.add_argument("--pathsim_va_strength", type=float, default=1.0,
                            help="Multiplier on predicted log variance shift; 0 recovers the standard path.")
        parser.add_argument("--pathsim_va_hidden_ch", type=int, default=48)
        parser.add_argument("--pathsim_va_num_bands", type=int, default=32)
        parser.add_argument("--pathsim_va_num_time_blocks", type=int, default=16)
        parser.add_argument("--pathsim_va_min_rho", type=float, default=0.5)
        parser.add_argument("--pathsim_va_max_rho", type=float, default=2.0)
        parser.add_argument("--pathsim_va_apply_in_validation", action="store_true",
                            help="Also use simulated states for batch-level valid_loss; file-level sampling stays unchanged.")

        # ===== Predictive residual-anchored diffusion refinement =====
        parser.add_argument("--residual_refine_enable", action="store_true",
                            help="Model the frontend correction residual with OUVE centered at a learned residual prior.")
        parser.add_argument("--residual_prior_type", type=str, default="direct",
                            choices=("direct", "predictive", "confidence_gate"),
                            help="Residual prior. 'predictive' is an alias of the direct complex predictor H_theta(Y_pre,Y_noisy); 'direct' is kept for checkpoint compatibility.")
        parser.add_argument("--residual_init_hidden_ch", type=int, default=48,
                            help="Hidden channels of the residual prior network.")
        parser.add_argument("--residual_init_num_blocks", type=int, default=4,
                            help="Number of dilated residual blocks in the residual prior network.")
        parser.add_argument("--residual_init_max_dilation", type=int, default=8,
                            help="Maximum dilation used by the residual prior network.")
        parser.add_argument("--residual_init_weight", type=float, default=0.25,
                            help="Outer weight of residual-prior supervision.")
        parser.add_argument("--residual_gate_init", type=float, default=0.10,
                            help="Initial confidence gate in (0,1) for confidence_gate prior.")
        parser.add_argument("--residual_gate_patch_freq", type=int, default=5,
                            help="Odd frequency-window size for the training oracle confidence map.")
        parser.add_argument("--residual_gate_patch_time", type=int, default=3,
                            help="Odd time-window size for the training oracle confidence map.")
        parser.add_argument("--residual_gate_floor_ratio", type=float, default=1e-4,
                            help="Relative R_obs-energy floor below which oracle gate is set to zero.")
        parser.add_argument("--residual_gate_oracle_weight", type=float, default=0.25,
                            help="Weight of energy-scaled oracle-gate supervision inside residual-prior loss.")
        parser.add_argument("--residual_init_joint_grad", action="store_true",
                            help="Allow the diffusion loss to update the residual prior network through the OUVE center. Off by default for stability.")
        parser.add_argument("--residual_refine_start_strength", type=float, default=1.0,
                            help="Default reverse start in [0,1]. For the new prior, 1.0 is recommended initially; can be overridden by inference --regen_strength.")
        parser.add_argument("--residual_anchor_schedule", type=str, default="constant",
                            choices=("constant", "power", "smoothstep"),
                            help="Time schedule lambda(t) for the learned residual anchor. constant reproduces the old Direct prior exactly; power/smoothstep fade the anchor toward zero near t=0.")
        parser.add_argument("--residual_anchor_strength", type=float, default=1.0,
                            help="Maximum learned-residual anchor strength lambda(T). 1.0 is the recommended first test.")
        parser.add_argument("--residual_anchor_power", type=float, default=1.0,
                            help="Shape exponent for power/smoothstep residual anchor schedules. Must be >= 1.")

        # ===== CTM-style distillation (Scheme-B) =====
        parser.add_argument("--ctm_distill", action="store_true", help="Enable CTM-style distillation training")
        parser.add_argument("--ctm_teacher_ckpt", type=str, default=None, help="Teacher ckpt path (SGMSE/SBVE checkpoint)")
        parser.add_argument("--ctm_k_teacher", type=int, default=4, help="Number of substeps for teacher partial solve t->u")
        parser.add_argument(
            "--ctm_u_mode",
            type=str,
            choices=("random", "fixed"),
            default="random",
            help="How to choose u in [s,t): random per-sample or fixed fraction.",
        )
        parser.add_argument("--ctm_u_frac", type=float, default=0.5, help="When ctm_u_mode=fixed, u = s + frac*(t-s)")
        parser.add_argument("--ctm_tmin", type=float, default=1e-4, help="Endpoint time used as '0' in CTM (avoid numerical issues)")
        parser.add_argument("--ctm_weight", type=float, default=1.0, help="Weight of CTM distillation loss")
        parser.add_argument("--ctm_mix_base", type=float, default=0.0, help="Mix in base diffusion loss (0 disables)")
        parser.add_argument("--ctm_prior_weight", type=float, default=0.0, help="Aux prior loss weight: ||x0_est - xD||")
        parser.add_argument("--ctm_lambda0", type=float, default=0.0, help="Guidance strength in teacher solver toward xD")
        parser.add_argument("--ctm_lambda_pow", type=float, default=2.0, help="Guidance schedule power for lambda(t) = lambda0*(1-t)^p")
        parser.add_argument("--ctm_time_l1", type=float, default=0.0, help="Optional time-domain L1 on endpoint (slow)")
        return parser

    def __init__(
        self,
        backbone: str,
        sde: str,
        train_type: str = "mode3",
        frontend: str = "semamba",
        frontend_root: Optional[str] = None,
        frontend_config: Optional[str] = None,
        frontend_ckpt: Optional[str] = None,
        frontend_module: Optional[str] = None,
        frontend_class: Optional[str] = None,
        frontend_state_dict_key: Optional[str] = None,
        frontend_trainable: bool = False,
        frontend_aux_loss: str = "none",
        weighting_denoiser_to_score: float = 0.5,
        # Frontend-relative no-regret / regret-aware score reweighting
        frontend_nr_strategy: str = "auxiliary",
        frontend_nr_reweight_strength: float = 1.0,
        frontend_nr_reweight_max: float = 4.0,
        frontend_nr_guidance_strength: float = 0.25,
        frontend_nr_guidance_temperature: float = 0.25,
        frontend_nr_guidance_tail_fraction: float = 1.0,
        frontend_nr_guidance_coverage_target: float = 0.0,
        rrsd_weight: float = 0.0,
        rrsd_hidden_ch: int = 48,
        rrsd_num_bands: int = 32,
        rrsd_num_time_blocks: int = 16,
        rrsd_init_alpha: float = 0.05,
        rrsd_benefit_weight: float = 4.0,
        rrsd_recon_weight: float = 0.10,
        rrsd_smooth_weight: float = 0.01,
        rrsd_oracle_eps: float = 1e-6,
        rrsd_teacher: str = "projection",
        rrsd_lookahead_k: int = 1,
        rrsd_lookahead_candidates: str = "0,0.25,0.5,0.75,1.0",
        rrsd_lookahead_dt: float = 0.0,
        rrsd_lookahead_chunk: int = 0,
        rrsd_future_weight: float = 1.0,
        rrsd_future_gate_cost: float = 1e-3,
        rrsd_use_gate: bool = False,
        rrsd_init_gate: float = 0.20,
        rrsd_gate_threshold: float = 0.01,
        rrsd_gate_local_threshold: float = 0.005,
        rrsd_gate_weight: float = 1.0,
        rrsd_strength_weight: float = 1.0,
        rrsd_gate_pos_weight: float = 2.0,
        rrsd_validate: bool = False,
        rrsd_freeze_base: bool = False,
        frontend_nr_weight: float = 0.0,
        frontend_nr_margin: float = 0.05,
        frontend_nr_t_max: float = 0.30,
        frontend_nr_eps: float = 1e-8,
        frontend_nr_mode: str = "global",
        frontend_nr_patch_freq: int = 5,
        frontend_nr_patch_time: int = 3,
        frontend_nr_topk: float = 0.20,
        frontend_nr_floor_ratio: float = 0.01,
        content_model_name: str = "microsoft/wavlm-base-plus",
        content_layer: int = 9,
        content_loss_type: str = "smooth_l1",
        final_content_weight: float = 0.0,
        selector_content_weight: float = 0.0,
        selector_oracle_weight: float = 0.0,
        selector_oracle_conf_weight: float = 1.0,
        selector_oracle_eps: float = 1e-8,
        # State-aware conditional anchor calibration
        anchor_calib_weight: float = 0.0,
        anchor_controller_type: str = "global_mlp",
        anchor_controller_hidden_ch: int = 64,
        anchor_num_bands: int = 32,
        anchor_k_max: float = 0.08,
        anchor_time_power: float = 1.0,
        anchor_band_smooth_weight: float = 0.0,
        anchor_loss_weight_c: float = 1.0,
        anchor_loss_weight_k: float = 1.0,
        anchor_loss_weight_recon: float = 0.1,
        # Anchor V2: fine-grained band-time x0 residual calibration
        anchor_v2_weight: float = 0.0,
        anchor_v2_hidden_ch: int = 48,
        anchor_v2_num_bands: int = 32,
        anchor_v2_num_time_blocks: int = 16,
        anchor_v2_a_max: float = 0.25,
        anchor_v2_b_max: float = 0.15,
        anchor_v2_time_power: float = 1.0,
        anchor_v2_loss_weight_a: float = 1.0,
        anchor_v2_loss_weight_b: float = 1.0,
        anchor_v2_loss_weight_recon: float = 0.1,
        anchor_v2_loss_weight_gain: float = 0.1,
        anchor_v2_smooth_weight: float = 0.01,
        anchor_v2_sparsity_weight: float = 0.001,
        anchor_v2_oracle_eps: float = 1e-6,
        anchor_v2_validate: bool = False,
        anchor_v2_single_shot_train: bool = False,
        anchor_v2_single_shot_ratio: float = 0.5,
        anchor_v2_single_shot_train_width: float = 0.10,
        anchor_v2_validate_single_shot: bool = False,
        # Path Simulation Version A
        pathsim_va_enable: bool = False,
        pathsim_va_ckpt: Optional[str] = None,
        pathsim_va_prob: float = 0.3,
        pathsim_va_strength: float = 1.0,
        pathsim_va_hidden_ch: int = 48,
        pathsim_va_num_bands: int = 32,
        pathsim_va_num_time_blocks: int = 16,
        pathsim_va_min_rho: float = 0.5,
        pathsim_va_max_rho: float = 2.0,
        pathsim_va_apply_in_validation: bool = False,
        # Predictive residual-anchored diffusion refinement
        residual_refine_enable: bool = False,
        residual_prior_type: str = "direct",
        residual_init_hidden_ch: int = 48,
        residual_init_num_blocks: int = 4,
        residual_init_max_dilation: int = 8,
        residual_init_weight: float = 0.25,
        residual_gate_init: float = 0.10,
        residual_gate_patch_freq: int = 5,
        residual_gate_patch_time: int = 3,
        residual_gate_floor_ratio: float = 1e-4,
        residual_gate_oracle_weight: float = 0.25,
        residual_init_joint_grad: bool = False,
        residual_refine_start_strength: float = 1.0,
        residual_anchor_schedule: str = "constant",
        residual_anchor_strength: float = 1.0,
        residual_anchor_power: float = 1.0,
        # CTM distillation
        ctm_distill: bool = False,
        ctm_teacher_ckpt: Optional[str] = None,
        ctm_k_teacher: int = 4,
        ctm_u_mode: str = "random",
        ctm_u_frac: float = 0.5,
        ctm_tmin: float = 1e-4,
        ctm_weight: float = 1.0,
        ctm_mix_base: float = 0.0,
        ctm_prior_weight: float = 0.0,
        ctm_lambda0: float = 0.0,
        ctm_lambda_pow: float = 2.0,
        ctm_time_l1: float = 0.0,
        data_module_cls=None,
        **kwargs,
    ):
        super().__init__(
            backbone=backbone,
            sde=sde,
            train_type=train_type,
            data_module_cls=data_module_cls,
            **kwargs,
        )

        self.frontend_name = str(frontend)
        self.frontend_aux_loss = str(frontend_aux_loss)
        self.weighting_denoiser_to_score = float(weighting_denoiser_to_score)

        # Frontend-relative no-regret. In score_reweight mode the detector is
        # detached and only redistributes the MAIN score-matching gradient across
        # TF regions; no extra score forward is needed.
        self.frontend_nr_strategy = str(frontend_nr_strategy).lower()
        self.frontend_nr_reweight_strength = float(frontend_nr_reweight_strength)
        self.frontend_nr_reweight_max = float(frontend_nr_reweight_max)
        self.frontend_nr_guidance_strength = float(frontend_nr_guidance_strength)
        self.frontend_nr_guidance_temperature = float(frontend_nr_guidance_temperature)
        self.frontend_nr_guidance_tail_fraction = float(frontend_nr_guidance_tail_fraction)
        self.frontend_nr_guidance_coverage_target = float(frontend_nr_guidance_coverage_target)

        # RRSD learns a deployable local routing coefficient alpha(t,f,tau) from
        # a clean-referenced projection oracle, while leaving the DSM target unchanged.
        self.rrsd_weight = float(rrsd_weight)
        self.rrsd_benefit_weight = float(rrsd_benefit_weight)
        self.rrsd_recon_weight = float(rrsd_recon_weight)
        self.rrsd_smooth_weight = float(rrsd_smooth_weight)
        self.rrsd_oracle_eps = float(rrsd_oracle_eps)
        self.rrsd_teacher = str(rrsd_teacher).strip().lower()
        self.rrsd_lookahead_k = int(rrsd_lookahead_k)
        self.rrsd_lookahead_candidates_raw = str(rrsd_lookahead_candidates)
        try:
            self.rrsd_lookahead_candidates = tuple(
                sorted(set(float(v.strip()) for v in self.rrsd_lookahead_candidates_raw.split(",") if v.strip()))
            )
        except Exception as e:
            raise ValueError(f"Invalid --rrsd_lookahead_candidates={rrsd_lookahead_candidates!r}") from e
        self.rrsd_lookahead_dt = float(rrsd_lookahead_dt)
        self.rrsd_lookahead_chunk = int(rrsd_lookahead_chunk)
        self.rrsd_future_weight = float(rrsd_future_weight)
        self.rrsd_future_gate_cost = float(rrsd_future_gate_cost)
        self.rrsd_use_gate = bool(rrsd_use_gate)
        self.rrsd_init_gate = float(rrsd_init_gate)
        self.rrsd_gate_threshold = float(rrsd_gate_threshold)
        self.rrsd_gate_local_threshold = float(rrsd_gate_local_threshold)
        self.rrsd_gate_weight = float(rrsd_gate_weight)
        self.rrsd_strength_weight = float(rrsd_strength_weight)
        self.rrsd_gate_pos_weight = float(rrsd_gate_pos_weight)
        self.rrsd_validate = bool(rrsd_validate)
        self.rrsd_freeze_base = bool(rrsd_freeze_base)
        if self.rrsd_weight < 0.0:
            raise ValueError("rrsd_weight must be >= 0.")
        if self.rrsd_benefit_weight < 0.0 or self.rrsd_recon_weight < 0.0 or self.rrsd_smooth_weight < 0.0:
            raise ValueError("RRSD loss weights must be >= 0.")
        if self.rrsd_gate_weight < 0.0 or self.rrsd_strength_weight < 0.0 or self.rrsd_gate_pos_weight <= 0.0:
            raise ValueError("RRSD gate/strength weights must be non-negative and gate_pos_weight > 0.")
        if self.rrsd_oracle_eps <= 0.0:
            raise ValueError("rrsd_oracle_eps must be > 0.")
        if self.rrsd_teacher not in {"projection", "kstep", "future_grad"}:
            raise ValueError("rrsd_teacher must be 'projection', 'kstep', or 'future_grad'.")
        if self.rrsd_lookahead_chunk < 0:
            raise ValueError("rrsd_lookahead_chunk must be >= 0.")
        if self.rrsd_future_weight < 0.0 or self.rrsd_future_gate_cost < 0.0:
            raise ValueError("rrsd_future_weight and rrsd_future_gate_cost must be >= 0.")
        if self.rrsd_lookahead_k < 1:
            raise ValueError("rrsd_lookahead_k must be >= 1.")
        if len(self.rrsd_lookahead_candidates) < 2 or 0.0 not in self.rrsd_lookahead_candidates:
            raise ValueError("rrsd_lookahead_candidates must contain 0 and at least one positive candidate.")
        if any(v < 0.0 or v > 1.0 for v in self.rrsd_lookahead_candidates):
            raise ValueError("rrsd_lookahead_candidates must lie in [0,1].")
        if self.rrsd_teacher in {"kstep", "future_grad"} and not self.rrsd_freeze_base:
            raise ValueError(
                f"--rrsd_teacher {self.rrsd_teacher} requires --rrsd_freeze_base so the diffusion model is a stable frozen evaluator."
            )

        # In frozen-base future-grad training, legacy ScoreModel contrastive loss
        # is attached to the *base* x0 estimate and therefore has no trainable path
        # to Q_phi. Re-purpose --contrastive_weight as a routed-future PLM loss:
        # it is evaluated on x0_next_route inside _rrsd_future_grad_loss.
        self.rrsd_route_contrastive = bool(
            self.rrsd_teacher == "future_grad"
            and self.rrsd_freeze_base
            and float(getattr(self, "contrastive_weight", 0.0)) > 0.0
        )
        if self.rrsd_route_contrastive:
            self.use_contrastive = False

        self.rrsd_router = None
        if self.rrsd_weight > 0.0:
            if self.sde.__class__.__name__ != "OUVESDE":
                raise ValueError("RRSD currently supports OUVESDE only.")
            if self.loss_type != "score_matching" or self.loss_weighting != "sigma^2":
                raise ValueError("RRSD currently requires score_matching with sigma^2 weighting.")
            self.rrsd_router = BandTimeReliabilityRouter(
                hidden_ch=int(rrsd_hidden_ch),
                num_bands=int(rrsd_num_bands),
                num_time_blocks=int(rrsd_num_time_blocks),
                init_alpha=float(rrsd_init_alpha),
                use_gate=self.rrsd_use_gate,
                init_gate=self.rrsd_init_gate,
            )

        self.frontend_nr_weight = float(frontend_nr_weight)
        self.frontend_nr_margin = float(frontend_nr_margin)
        self.frontend_nr_t_max = float(frontend_nr_t_max)
        self.frontend_nr_eps = float(frontend_nr_eps)
        self.frontend_nr_mode = str(frontend_nr_mode).lower()
        self.frontend_nr_patch_freq = int(frontend_nr_patch_freq)
        self.frontend_nr_patch_time = int(frontend_nr_patch_time)
        self.frontend_nr_topk = float(frontend_nr_topk)
        self.frontend_nr_floor_ratio = float(frontend_nr_floor_ratio)
        if self.frontend_nr_strategy not in {"auxiliary", "score_reweight", "adaptive_score_guidance"}:
            raise ValueError(
                "frontend_nr_strategy must be one of "
                "{'auxiliary', 'score_reweight', 'adaptive_score_guidance'}."
            )
        if self.frontend_nr_reweight_strength < 0.0:
            raise ValueError("frontend_nr_reweight_strength must be >= 0.")
        if self.frontend_nr_reweight_max < 1.0:
            raise ValueError("frontend_nr_reweight_max must be >= 1.")
        if self.frontend_nr_guidance_strength < 0.0:
            raise ValueError("frontend_nr_guidance_strength must be >= 0.")
        if self.frontend_nr_guidance_temperature <= 0.0:
            raise ValueError("frontend_nr_guidance_temperature must be > 0.")
        if not (0.0 < self.frontend_nr_guidance_tail_fraction <= 1.0):
            raise ValueError("frontend_nr_guidance_tail_fraction must be in (0, 1].")
        if self.frontend_nr_guidance_coverage_target < 0.0:
            raise ValueError("frontend_nr_guidance_coverage_target must be >= 0.")
        if self.frontend_nr_weight < 0.0:
            raise ValueError("frontend_nr_weight must be >= 0.")
        if self.frontend_nr_margin < 0.0:
            raise ValueError("frontend_nr_margin must be >= 0.")
        if self.frontend_nr_eps <= 0.0:
            raise ValueError("frontend_nr_eps must be > 0.")
        if self.frontend_nr_mode not in {"global", "patch"}:
            raise ValueError("frontend_nr_mode must be one of {'global', 'patch'}.")
        if self.frontend_nr_strategy in {"score_reweight", "adaptive_score_guidance"} and self.frontend_nr_mode != "patch":
            raise ValueError(
                "frontend_nr_strategy in {'score_reweight', 'adaptive_score_guidance'} "
                "requires --frontend_nr_mode patch."
            )
        if self.frontend_nr_patch_freq <= 0 or self.frontend_nr_patch_freq % 2 == 0:
            raise ValueError("frontend_nr_patch_freq must be a positive odd integer.")
        if self.frontend_nr_patch_time <= 0 or self.frontend_nr_patch_time % 2 == 0:
            raise ValueError("frontend_nr_patch_time must be a positive odd integer.")
        if not (0.0 < self.frontend_nr_topk <= 1.0):
            raise ValueError("frontend_nr_topk must be in (0, 1].")
        if self.frontend_nr_floor_ratio < 0.0:
            raise ValueError("frontend_nr_floor_ratio must be >= 0.")
        nr_enabled = (
            (self.frontend_nr_strategy == "auxiliary" and self.frontend_nr_weight > 0.0)
            or (self.frontend_nr_strategy == "score_reweight" and self.frontend_nr_reweight_strength > 0.0)
            or (
                self.frontend_nr_strategy == "adaptive_score_guidance"
                and self.frontend_nr_guidance_strength > 0.0
            )
        )
        if nr_enabled:
            if self.sde.__class__.__name__ != "OUVESDE":
                raise ValueError("frontend no-regret loss currently supports --sde ouve only.")
            if self.loss_type != "score_matching" or self.loss_weighting != "sigma^2":
                raise ValueError(
                    "frontend no-regret loss currently supports "
                    "--loss_type score_matching --loss_weighting sigma^2 only."
                )
            if not float(getattr(self, "t_eps", 0.03)) <= self.frontend_nr_t_max <= float(self.sde.T):
                raise ValueError(
                    "frontend_nr_t_max must be in [t_eps, sde.T], got "
                    f"{self.frontend_nr_t_max}."
                )
        # 内容对齐loss参数
        self.content_model_name = str(content_model_name)
        self.content_layer = int(content_layer)
        self.content_loss_type = str(content_loss_type)
        self.final_content_weight = float(final_content_weight)
        self.selector_content_weight = float(selector_content_weight)
        self.selector_oracle_weight = float(selector_oracle_weight)
        self.selector_oracle_conf_weight = float(selector_oracle_conf_weight)
        self.selector_oracle_eps = float(selector_oracle_eps)

        # State-aware conditional anchor calibration.
        # This branch is sampler-side: it learns how much the OUVE reverse mean
        # should be shifted toward an adaptive anchor during sampling.
        self.anchor_calib_weight = float(anchor_calib_weight)
        self.anchor_controller_type = str(anchor_controller_type)
        self.anchor_num_bands = int(anchor_num_bands)
        self.anchor_band_smooth_weight = float(anchor_band_smooth_weight)
        self.anchor_loss_weight_c = float(anchor_loss_weight_c)
        self.anchor_loss_weight_k = float(anchor_loss_weight_k)
        self.anchor_loss_weight_recon = float(anchor_loss_weight_recon)
        self.anchor_controller = None
        if self.anchor_calib_weight > 0.0:
            if self.anchor_controller_type == "band_mlp":
                self.anchor_controller = StateAwareBandAnchorController(
                    hidden_ch=int(anchor_controller_hidden_ch),
                    k_max=float(anchor_k_max),
                    time_power=float(anchor_time_power),
                    num_bands=int(anchor_num_bands),
                )
            else:
                self.anchor_controller = StateAwareAnchorController(
                    hidden_ch=int(anchor_controller_hidden_ch),
                    k_max=float(anchor_k_max),
                    time_power=float(anchor_time_power),
                )

        # Anchor V2 is an independent branch.  It leaves the original anchor
        # controller and all of its checkpoint keys untouched.
        self.anchor_v2_weight = float(anchor_v2_weight)
        self.anchor_v2_loss_weight_a = float(anchor_v2_loss_weight_a)
        self.anchor_v2_loss_weight_b = float(anchor_v2_loss_weight_b)
        self.anchor_v2_loss_weight_recon = float(anchor_v2_loss_weight_recon)
        self.anchor_v2_loss_weight_gain = float(anchor_v2_loss_weight_gain)
        self.anchor_v2_smooth_weight = float(anchor_v2_smooth_weight)
        self.anchor_v2_sparsity_weight = float(anchor_v2_sparsity_weight)
        self.anchor_v2_oracle_eps = float(anchor_v2_oracle_eps)
        self.anchor_v2_validate = bool(anchor_v2_validate)
        self.anchor_v2_single_shot_train = bool(anchor_v2_single_shot_train)
        self.anchor_v2_single_shot_ratio = float(anchor_v2_single_shot_ratio)
        self.anchor_v2_single_shot_train_width = float(anchor_v2_single_shot_train_width)
        self.anchor_v2_validate_single_shot = bool(anchor_v2_validate_single_shot)
        if not 0.0 <= self.anchor_v2_single_shot_ratio <= 1.0:
            raise ValueError("anchor_v2_single_shot_ratio must be in [0,1].")
        if not 0.0 <= self.anchor_v2_single_shot_train_width <= 1.0:
            raise ValueError("anchor_v2_single_shot_train_width must be in [0,1].")
        self.anchor_v2_controller = None
        if self.anchor_v2_weight > 0.0:
            self.anchor_v2_controller = BandTimeAnchorV2Controller(
                hidden_ch=int(anchor_v2_hidden_ch),
                num_bands=int(anchor_v2_num_bands),
                num_time_blocks=int(anchor_v2_num_time_blocks),
                a_max=float(anchor_v2_a_max),
                b_max=float(anchor_v2_b_max),
                time_power=float(anchor_v2_time_power),
            )

        # PathSim V-A is trained independently and remains frozen here.  It
        # changes only the training-state variance, never the inference sampler.
        self.pathsim_va_enable = bool(pathsim_va_enable)
        self.pathsim_va_prob = float(pathsim_va_prob)
        self.pathsim_va_strength = float(pathsim_va_strength)
        self.pathsim_va_apply_in_validation = bool(pathsim_va_apply_in_validation)
        if not 0.0 <= self.pathsim_va_prob <= 1.0:
            raise ValueError(f"pathsim_va_prob must be in [0,1], got {self.pathsim_va_prob}")
        self.pathsim_va_model = None
        if self.pathsim_va_enable:
            if self.loss_type != "score_matching" or self.loss_weighting != "sigma^2":
                raise ValueError(
                    "PathSim V-A currently supports --loss_type score_matching "
                    "--loss_weighting sigma^2 only."
                )
            if pathsim_va_ckpt:
                model_va, _ = load_pathsim_va_checkpoint(pathsim_va_ckpt, map_location="cpu")
                self.pathsim_va_model = model_va
            else:
                warnings.warn(
                    "--pathsim_va_enable was set without --pathsim_va_ckpt; "
                    "using a zero-initialized rho=1 estimator until checkpoint weights are loaded."
                )
                self.pathsim_va_model = BandTimeVariancePathEstimator(
                    hidden_ch=int(pathsim_va_hidden_ch),
                    num_bands=int(pathsim_va_num_bands),
                    num_time_blocks=int(pathsim_va_num_time_blocks),
                    min_rho=float(pathsim_va_min_rho),
                    max_rho=float(pathsim_va_max_rho),
                )
            self.pathsim_va_model.eval()
            for p in self.pathsim_va_model.parameters():
                p.requires_grad = False

        # Learned residual-anchored diffusion refinement.  The predictive
        # prior estimates R_theta = H_theta(Y_pre, Y_noisy).  OUVE can then use
        # a time-dependent anchor A(t)=lambda(t) R_theta in residual coordinates
        # (equivalently Y_pre + lambda(t) R_theta in clean-spectrum coordinates).
        # By default the score loss does not backpropagate into the prior network.
        self.residual_refine_enable = bool(residual_refine_enable)
        if self.rrsd_weight > 0.0 and self.residual_refine_enable:
            raise ValueError(
                "RRSD is defined for the full-speech frontend/diffusion path only; "
                "disable --residual_refine_enable for this experiment."
            )
        self.residual_prior_type = str(residual_prior_type).strip().lower()
        # 'predictive' is a paper/CLI alias; the module/checkpoint layout is
        # intentionally identical to the historical 'direct' predictor.
        if self.residual_prior_type == "predictive":
            self.residual_prior_type = "direct"
        self.residual_init_weight = float(residual_init_weight)
        self.residual_gate_init = float(residual_gate_init)
        self.residual_gate_patch_freq = int(residual_gate_patch_freq)
        self.residual_gate_patch_time = int(residual_gate_patch_time)
        self.residual_gate_floor_ratio = float(residual_gate_floor_ratio)
        self.residual_gate_oracle_weight = float(residual_gate_oracle_weight)
        self.residual_init_joint_grad = bool(residual_init_joint_grad)
        self.residual_refine_start_strength = float(residual_refine_start_strength)
        self.residual_anchor_schedule = str(residual_anchor_schedule).strip().lower()
        self.residual_anchor_strength = float(residual_anchor_strength)
        self.residual_anchor_power = float(residual_anchor_power)
        if self.residual_prior_type not in {"direct", "confidence_gate"}:
            raise ValueError("residual_prior_type must be 'direct' or 'confidence_gate'.")
        if self.residual_init_weight < 0.0:
            raise ValueError("residual_init_weight must be >= 0.")
        if not 0.0 < self.residual_gate_init < 1.0:
            raise ValueError("residual_gate_init must be in (0,1).")
        if self.residual_gate_patch_freq < 1 or self.residual_gate_patch_freq % 2 == 0:
            raise ValueError("residual_gate_patch_freq must be a positive odd integer.")
        if self.residual_gate_patch_time < 1 or self.residual_gate_patch_time % 2 == 0:
            raise ValueError("residual_gate_patch_time must be a positive odd integer.")
        if self.residual_gate_floor_ratio < 0.0:
            raise ValueError("residual_gate_floor_ratio must be >= 0.")
        if self.residual_gate_oracle_weight < 0.0:
            raise ValueError("residual_gate_oracle_weight must be >= 0.")
        if not 0.0 <= self.residual_refine_start_strength <= 1.0:
            raise ValueError("residual_refine_start_strength must be in [0,1].")
        if self.residual_anchor_schedule not in {"constant", "power", "smoothstep"}:
            raise ValueError("residual_anchor_schedule must be one of {'constant','power','smoothstep'}.")
        if self.residual_anchor_strength < 0.0:
            raise ValueError("residual_anchor_strength must be >= 0.")
        if self.residual_anchor_power < 1.0:
            raise ValueError("residual_anchor_power must be >= 1.")
        self.residual_initializer = None
        if self.residual_refine_enable:
            if self.train_type != "mode2":
                raise ValueError("--residual_refine_enable currently requires --train_type mode2")
            if self.sde.__class__.__name__ != "OUVESDE":
                raise ValueError("--residual_refine_enable currently requires --sde ouve")
            if self.loss_type != "score_matching" or self.loss_weighting != "sigma^2":
                raise ValueError(
                    "--residual_refine_enable currently requires "
                    "--loss_type score_matching --loss_weighting sigma^2"
                )
            if self.anchor_calib_weight > 0.0 or self.anchor_v2_weight > 0.0:
                raise ValueError("Disable Anchor V1/V2 for the first residual-refinement experiment.")
            if self.pathsim_va_enable:
                raise ValueError(
                    "Predictive residual refinement and PathSim V-A use different state semantics; "
                    "disable --pathsim_va_enable when --residual_refine_enable is active."
                )
            self.residual_initializer = build_residual_prior_predictor(
                prior_type=self.residual_prior_type,
                hidden_ch=int(residual_init_hidden_ch),
                num_blocks=int(residual_init_num_blocks),
                max_dilation=int(residual_init_max_dilation),
                initial_gate=self.residual_gate_init,
            )
            if not hasattr(self.sde, "configure_residual_anchor"):
                raise ValueError("The selected OUVE implementation does not support residual anchor schedules.")
            self.sde.configure_residual_anchor(
                enabled=True,
                schedule=self.residual_anchor_schedule,
                strength=self.residual_anchor_strength,
                power=self.residual_anchor_power,
            )

        # ScoreModel.save_hyperparameters() runs in the base constructor before
        # these subclass-only options are normalized, so explicitly persist them.
        self.hparams["frontend_nr_strategy"] = self.frontend_nr_strategy
        self.hparams["frontend_nr_reweight_strength"] = self.frontend_nr_reweight_strength
        self.hparams["frontend_nr_reweight_max"] = self.frontend_nr_reweight_max
        self.hparams["frontend_nr_guidance_strength"] = self.frontend_nr_guidance_strength
        self.hparams["frontend_nr_guidance_temperature"] = self.frontend_nr_guidance_temperature
        self.hparams["frontend_nr_guidance_tail_fraction"] = self.frontend_nr_guidance_tail_fraction
        self.hparams["frontend_nr_guidance_coverage_target"] = self.frontend_nr_guidance_coverage_target
        self.hparams["rrsd_weight"] = self.rrsd_weight
        self.hparams["rrsd_hidden_ch"] = int(rrsd_hidden_ch)
        self.hparams["rrsd_num_bands"] = int(rrsd_num_bands)
        self.hparams["rrsd_num_time_blocks"] = int(rrsd_num_time_blocks)
        self.hparams["rrsd_init_alpha"] = float(rrsd_init_alpha)
        self.hparams["rrsd_benefit_weight"] = self.rrsd_benefit_weight
        self.hparams["rrsd_recon_weight"] = self.rrsd_recon_weight
        self.hparams["rrsd_smooth_weight"] = self.rrsd_smooth_weight
        self.hparams["rrsd_oracle_eps"] = self.rrsd_oracle_eps
        self.hparams["rrsd_teacher"] = self.rrsd_teacher
        self.hparams["rrsd_lookahead_k"] = self.rrsd_lookahead_k
        self.hparams["rrsd_lookahead_candidates"] = self.rrsd_lookahead_candidates_raw
        self.hparams["rrsd_lookahead_dt"] = self.rrsd_lookahead_dt
        self.hparams["rrsd_lookahead_chunk"] = self.rrsd_lookahead_chunk
        self.hparams["rrsd_future_weight"] = self.rrsd_future_weight
        self.hparams["rrsd_future_gate_cost"] = self.rrsd_future_gate_cost
        self.hparams["rrsd_use_gate"] = self.rrsd_use_gate
        self.hparams["rrsd_init_gate"] = self.rrsd_init_gate
        self.hparams["rrsd_gate_threshold"] = self.rrsd_gate_threshold
        self.hparams["rrsd_gate_local_threshold"] = self.rrsd_gate_local_threshold
        self.hparams["rrsd_gate_weight"] = self.rrsd_gate_weight
        self.hparams["rrsd_strength_weight"] = self.rrsd_strength_weight
        self.hparams["rrsd_gate_pos_weight"] = self.rrsd_gate_pos_weight
        self.hparams["rrsd_validate"] = self.rrsd_validate
        self.hparams["rrsd_freeze_base"] = self.rrsd_freeze_base
        self.hparams["frontend_nr_weight"] = self.frontend_nr_weight
        self.hparams["frontend_nr_margin"] = self.frontend_nr_margin
        self.hparams["frontend_nr_t_max"] = self.frontend_nr_t_max
        self.hparams["frontend_nr_mode"] = self.frontend_nr_mode
        self.hparams["frontend_nr_patch_freq"] = self.frontend_nr_patch_freq
        self.hparams["frontend_nr_patch_time"] = self.frontend_nr_patch_time
        self.hparams["frontend_nr_topk"] = self.frontend_nr_topk
        self.hparams["frontend_nr_floor_ratio"] = self.frontend_nr_floor_ratio
        self.hparams["residual_refine_enable"] = self.residual_refine_enable
        self.hparams["residual_prior_type"] = self.residual_prior_type
        self.hparams["residual_init_hidden_ch"] = int(residual_init_hidden_ch)
        self.hparams["residual_init_num_blocks"] = int(residual_init_num_blocks)
        self.hparams["residual_init_max_dilation"] = int(residual_init_max_dilation)
        self.hparams["residual_init_weight"] = self.residual_init_weight
        self.hparams["residual_gate_init"] = self.residual_gate_init
        self.hparams["residual_gate_patch_freq"] = self.residual_gate_patch_freq
        self.hparams["residual_gate_patch_time"] = self.residual_gate_patch_time
        self.hparams["residual_gate_floor_ratio"] = self.residual_gate_floor_ratio
        self.hparams["residual_gate_oracle_weight"] = self.residual_gate_oracle_weight
        self.hparams["residual_init_joint_grad"] = self.residual_init_joint_grad
        self.hparams["residual_refine_start_strength"] = self.residual_refine_start_strength
        self.hparams["residual_anchor_schedule"] = self.residual_anchor_schedule
        self.hparams["residual_anchor_strength"] = self.residual_anchor_strength
        self.hparams["residual_anchor_power"] = self.residual_anchor_power

        self.use_content_loss = (self.final_content_weight > 0.0) or (self.selector_content_weight > 0.0)
        self.content_model = None
        if self.use_content_loss:
            try:
                self.content_model = WavLMModel.from_pretrained(self.content_model_name)
                self.content_model.eval()
                for p in self.content_model.parameters():
                    p.requires_grad = False
            except Exception as e:
                warnings.warn(
                    f"Failed to load content model '{self.content_model_name}': {e}. "
                    "Content loss will be disabled."
                )
                self.content_model = None
                self.use_content_loss = False

        # CTM distillation cfg
        self.ctm_distill = bool(ctm_distill)
        self.ctm_teacher_ckpt = ctm_teacher_ckpt
        self.ctm_k_teacher = int(ctm_k_teacher)
        self.ctm_u_mode = str(ctm_u_mode)
        self.ctm_u_frac = float(ctm_u_frac)
        self.ctm_tmin = float(ctm_tmin)
        self.ctm_weight = float(ctm_weight)
        self.ctm_mix_base = float(ctm_mix_base)
        self.ctm_prior_weight = float(ctm_prior_weight)
        self.ctm_lambda0 = float(ctm_lambda0)
        self.ctm_lambda_pow = float(ctm_lambda_pow)
        self.ctm_time_l1 = float(ctm_time_l1)
        if self.residual_refine_enable and self.ctm_distill:
            raise ValueError("Residual refinement and CTM distillation cannot be enabled together.")

        # Teacher backbone (frozen) for CTM distillation
        self.teacher_dnn = None
        self.teacher_ema = None
        if self.ctm_distill:
            if not self.ctm_teacher_ckpt:
                raise ValueError("--ctm_distill requires --ctm_teacher_ckpt")
            self._init_teacher_from_ckpt(self.ctm_teacher_ckpt)

        self.frontend = build_frontend(
            frontend_name=self.frontend_name,
            frontend_root=frontend_root,
            frontend_config=frontend_config,
            frontend_ckpt=frontend_ckpt,
            frontend_module=frontend_module,
            frontend_class=frontend_class,
            frontend_state_dict_key=frontend_state_dict_key,
            **kwargs,
        )

        self.frontend_trainable = bool(frontend_trainable)
        if not self.frontend_trainable:
            for p in self.frontend.parameters():
                p.requires_grad = False
            self.frontend.eval()

        if self.rrsd_freeze_base:
            if self.rrsd_router is None:
                raise ValueError("--rrsd_freeze_base requires --rrsd_weight > 0 so a router exists.")
            for name, p in self.named_parameters():
                p.requires_grad = name.startswith("rrsd_router.")

        # Rebuild EMA after frontend creation so EMA tracks the full trainable model
        self._ensure_ema_is_compatible(force_rebuild=True)

    # ------------------------- CTM distillation helpers -------------------------
    def _init_teacher_from_ckpt(self, ckpt_path: str) -> None:
        """Create a frozen teacher_dnn from a Lightning checkpoint.

        We load dnn.* weights; if an EMA state is present, we copy EMA weights into teacher_dnn.
        """
        try:
            ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        except TypeError:
            ckpt = torch.load(ckpt_path, map_location="cpu")
        state = ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt
        # copy architecture
        self.teacher_dnn = copy.deepcopy(self.dnn).eval()
        for p in self.teacher_dnn.parameters():
            p.requires_grad = False

        # extract dnn weights
        dnn_state = {}
        if isinstance(state, dict):
            for k, v in state.items():
                if k.startswith("dnn."):
                    dnn_state[k[len("dnn."):]] = v
        if dnn_state:
            self.teacher_dnn.load_state_dict(dnn_state, strict=False)

        # if checkpoint contains EMA, prefer EMA weights
        if isinstance(ckpt, dict) and "ema" in ckpt:
            try:
                from torch_ema import ExponentialMovingAverage

                self.teacher_ema = ExponentialMovingAverage(self.teacher_dnn.parameters(), decay=self.ema_decay)
                self.teacher_ema.load_state_dict(ckpt["ema"])
                self.teacher_ema.copy_to(self.teacher_dnn.parameters())
            except Exception as e:
                print(f"[CTM] Warning: failed to restore teacher EMA weights: {e}")

    @contextmanager
    def _use_target_ema(self):
        """Temporarily swap student weights to EMA (target network) weights."""
        ema_params = self._get_ema_parameters()
        self.ema.store(ema_params)
        self.ema.copy_to(ema_params)
        try:
            yield
        finally:
            self.ema.restore(ema_params)

    def _forward_with_dnn(self, dnn: torch.nn.Module, x_t, y, t, y_cond=None):
        """Same as ScoreModel.forward but using the provided dnn module.

        NOTE: Tailored for backbone='ncsnpp_v2' and SBVE+EDM (x0 prediction).
        """
        y_cond = y if y_cond is None else y_cond
        if self.backbone != "ncsnpp_v2":
            raise NotImplementedError("CTM distill currently supports backbone=ncsnpp_v2 only")
        if self.loss_weighting != "edm" or self.loss_type not in ("denoiser", "data_prediction"):
            raise ValueError("CTM distill expects SBVE x0-prediction setup: --loss_type denoiser --loss_weighting edm")
        if self.sde.__class__.__name__ != "SBVESDE":
            raise ValueError("CTM distill currently implemented for sde=sbve only")

        # EDM coordinate transform
        w_xt, w_yt = self.sde.weights(t)
        x_in = (x_t - w_yt * y) / (w_xt + 1e-12)

        # network forward
        y_extra = None
        if self.train_type == "mode2" and self.mode2_include_noisy_main and y_cond is not None:
            y_extra = self._c_in(t) * y_cond
        Fnet = dnn(self._c_in(t) * x_in, self._c_in(t) * y, t, y_cond=self._c_in(t) * y_cond, y_extra=y_extra)

        # optional scaling
        if self.network_scaling == "1/sigma":
            std = self._sigma_for_precond(t)
            Fnet = Fnet / std[:, None, None, None]
        elif self.network_scaling == "1/t":
            Fnet = Fnet / t[:, None, None, None]

        # SBVE expects x0_hat
        x_hat = self._c_skip(t) * x_in + self._c_out(t) * Fnet
        return x_hat

    def _sbve_ode_transition(self, x_prev, y, t_prev, t_cur, x0_hat, xD=None):
        """One deterministic SBVE ODE update from t_prev -> t_cur.

        Matches the update in sampling.get_sb_sampler(..., sampler_type='ode').
        Optionally applies Scheme-B guidance toward xD after the SB update.
        """
        sde: SBVESDE = self.sde
        # schedules for prev & cur times
        sigma_prev, sigma_T, sigma_bar_prev, alpha_prev, alpha_T, _ = sde._sigmas_alphas(t_prev)
        sigma_t, sigma_T2, sigma_bart, alpha_t, alpha_T2, _ = sde._sigmas_alphas(t_cur)
        sigma_T = sigma_T2
        alpha_T = alpha_T2

        weight_prev = alpha_t * sigma_t * sigma_bart / (alpha_prev * sigma_prev * sigma_bar_prev + sde.eps)
        weight_estimate = (
            alpha_t
            / (sigma_T**2 + sde.eps)
            * (sigma_bart**2 - sigma_bar_prev * sigma_t * sigma_bart / (sigma_prev + sde.eps))
        )
        weight_prior_mean = (
            alpha_t
            / (alpha_T * sigma_T**2 + sde.eps)
            * (sigma_t**2 - sigma_prev * sigma_t * sigma_bart / (sigma_bar_prev + sde.eps))
        )

        weight_prev = weight_prev[:, None, None, None]
        weight_estimate = weight_estimate[:, None, None, None]
        weight_prior_mean = weight_prior_mean[:, None, None, None]

        x_new = weight_prev * x_prev + weight_estimate * x0_hat + weight_prior_mean * y

        # Scheme-B guidance: pull toward discriminative estimate xD
        if xD is not None and self.ctm_lambda0 > 0.0:
            dt = (t_prev - t_cur).clamp_min(0.0)
            lam = self.ctm_lambda0 * torch.pow((1.0 - t_cur).clamp_min(0.0), self.ctm_lambda_pow)
            gamma = 1.0 - torch.exp(-(lam * dt).clamp_min(0.0))
            x_new = x_new + gamma[:, None, None, None] * (xD - x_new)

        return x_new

    def _teacher_partial_solve(self, x_t, y, t, u, y_cond=None, xD=None):
        """Teacher solver from t -> u with K substeps, optionally guided toward xD."""
        if self.teacher_dnn is None:
            raise RuntimeError("Teacher not initialized")
        K = max(int(self.ctm_k_teacher), 1)
        x = x_t
        t_prev = t
        for k in range(K):
            frac = (k + 1) / K
            t_cur = t - (t - u) * frac
            with torch.no_grad():
                x0_hat = self._forward_with_dnn(self.teacher_dnn, x, y, t_cur, y_cond=y_cond)
            x = self._sbve_ode_transition(x, y, t_prev, t_cur, x0_hat, xD=xD)
            t_prev = t_cur
        return x

    def _G_apply(self, which: str, x_prev, y, t_prev, t_cur, y_cond=None, xD=None):
        """Apply jump operator from t_prev -> t_cur using student/target/teacher model."""
        if which not in ("student", "target", "teacher"):
            raise ValueError("which must be one of: student/target/teacher")

        if which == "teacher":
            with torch.no_grad():
                x0_hat = self._forward_with_dnn(self.teacher_dnn, x_prev, y, t_cur, y_cond=y_cond)
            return self._sbve_ode_transition(x_prev, y, t_prev, t_cur, x0_hat, xD=xD)

        if which == "target":
            with torch.no_grad(), self._use_target_ema():
                x0_hat = self._forward_with_dnn(self.dnn, x_prev, y, t_cur, y_cond=y_cond)
                return self._sbve_ode_transition(x_prev, y, t_prev, t_cur, x0_hat, xD=None)

        # student
        x0_hat = self._forward_with_dnn(self.dnn, x_prev, y, t_cur, y_cond=y_cond)
        return self._sbve_ode_transition(x_prev, y, t_prev, t_cur, x0_hat, xD=None)


    def _maybe_apply_pathsim_va(self, x, y, y_cond, t, mean, std, z, x_t_standard):
        """Replace selected analytical marginal states by variance-shifted states.

        The frozen estimator predicts rho(f,tau,t).  For selected samples:
            x_t = mean + std * rho * z
        and the score target uses the matching local sigma_eff=std*rho.
        """
        if (not self.pathsim_va_enable) or self.pathsim_va_model is None:
            return x_t_standard, None, {}
        if (not self.training) and (not self.pathsim_va_apply_in_validation):
            return x_t_standard, None, {}

        if self.train_type == "mode2":
            y_pre, y_noisy = y, y_cond
        elif self.train_type == "mode3":
            y_pre, y_noisy = y_cond, y
        else:
            return x_t_standard, None, {}
        if y_pre is None or y_noisy is None:
            return x_t_standard, None, {}

        self.pathsim_va_model.eval()
        with torch.no_grad():
            pred = self.pathsim_va_model(
                y_pre.detach(), y_noisy.detach(), t.detach(),
                t_max=float(self.sde.T), return_full=True,
                full_size=(int(x.shape[-2]), int(x.shape[-1])),
            )
            log_var = pred["log_var_full"].to(device=x.device, dtype=x.real.dtype)
            log_var = float(self.pathsim_va_strength) * log_var
            rho = torch.exp(0.5 * log_var)

            B = x.shape[0]
            if self.pathsim_va_prob >= 1.0:
                use_sim = torch.ones((B, 1, 1, 1), device=x.device, dtype=torch.bool)
            elif self.pathsim_va_prob <= 0.0:
                use_sim = torch.zeros((B, 1, 1, 1), device=x.device, dtype=torch.bool)
            else:
                use_sim = (torch.rand((B, 1, 1, 1), device=x.device) < self.pathsim_va_prob)
            rho_eff = torch.where(use_sim, rho, torch.ones_like(rho))

            std_full = std.to(device=x.device, dtype=x.real.dtype).view(B, 1, 1, 1)
            sigma_eff = std_full * rho_eff
            x_t = mean + sigma_eff * z

            stats = {
                "pathsim_va_use_fraction": use_sim.float().mean(),
                "pathsim_va_rho_mean": rho_eff.mean(),
                "pathsim_va_rho_sim_mean": (
                    (rho * use_sim.float()).sum() / use_sim.float().expand_as(rho).sum().clamp_min(1.0)
                ),
                "pathsim_va_log_var_abs_mean": log_var.abs().mean(),
                "pathsim_va_sigma_ratio_mean": rho_eff.mean(),
            }
        return x_t, sigma_eff, stats

    def configure_optimizers(self):
        params = [p for p in self.parameters() if p.requires_grad]
        optimizer = torch.optim.Adam(params, lr=self.lr)
        return optimizer
    def train(self, mode: bool = True, no_ema: bool = False):
        out = super().train(mode, no_ema=no_ema)
        if mode and bool(getattr(self, "rrsd_freeze_base", False)):
            # Keep the frozen teacher system deterministic while Q_phi learns.
            if hasattr(self, "dnn") and self.dnn is not None:
                self.dnn.eval()
            if hasattr(self, "frontend") and self.frontend is not None:
                self.frontend.eval()
            if getattr(self, "residual_initializer", None) is not None:
                self.residual_initializer.eval()
            if getattr(self, "rrsd_router", None) is not None:
                self.rrsd_router.train(True)
        return out
    def on_fit_start(self):
        # Make sure teacher model is on the same device
        super().on_fit_start()
        if self.teacher_dnn is not None:
            self.teacher_dnn.to(self.device)

    def _frontend_forward(self, y_noisy: torch.Tensor) -> torch.Tensor:
        """统一前端前向：输入/输出均为波形 `[B,1,T]`。"""
        return self.frontend(y_noisy)

    def _semamba_forward(self, y_noisy: torch.Tensor) -> torch.Tensor:
        """Backward-compatible alias for older debug/inference scripts."""
        return self._frontend_forward(y_noisy)

    def _sgmse_stft(self, wav: torch.Tensor) -> torch.Tensor:
        """torch.stft 只接受 1D([T]) 或 2D([B,T])。
            dataloader 常给 [B,1,T]，这里做一次 squeeze。
        """
        if wav.dim() == 3 and wav.size(1) == 1:
            wav = wav[:, 0, :]          # [B,T]
        elif wav.dim() == 1:
            wav = wav.unsqueeze(0)      # [1,T]
        elif wav.dim() != 2:
            raise RuntimeError(f"_sgmse_stft expects [T], [B,T] or [B,1,T], got {tuple(wav.shape)}")
        # 先把 [B,1,T] squeeze 成 [B,T] 做 stft，再 unsqueeze(1) 回 [B,1,F,T]
        wav2 = wav.squeeze(1) if wav.dim() == 3 and wav.size(1) == 1 else wav

        window = self.data_module.window.to(wav.device)
        spec = torch.stft(# [B,F,T]
            wav2,
            n_fft=self.data_module.n_fft,
            hop_length=self.data_module.hop_length,
            window=window,
            center=True,
            return_complex=True,
        )
        spec = spec.unsqueeze(1) # [B,1,F,T]
        return self.data_module.spec_fwd(spec)

    def _print_tensor_stats(self, name, x, batch_idx, force=False):
        """打印 tensor 的统计信息，用于调试"""
        if not force and batch_idx % 100 != 0:  # 每100个batch打印一次
            return
        x_det = x.detach()
        # 处理复数类型 (ComplexFloat/ComplexDouble)
        if torch.is_complex(x_det):
            x_det = x_det.abs()  # 取模转为实数
        print(f"[DEBUG:{batch_idx}] {name}: shape={tuple(x_det.shape)}, "
              f"dtype={x_det.dtype}, device={x_det.device}, "
              f"mean={x_det.mean().item():.6f}, std={x_det.std().item():.6f}, "
              f"min={x_det.min().item():.6f}, max={x_det.max().item():.6f}, "
              f"nan={torch.isnan(x_det).any().item()}, inf={torch.isinf(x_det).any().item()}")

    def _compute_frontend_aux_loss(
        self,
        x_wav: torch.Tensor,
        y_pre: torch.Tensor,
        X: torch.Tensor,
        Y_pre: torch.Tensor,
    ) -> torch.Tensor:
        loss_type = str(self.frontend_aux_loss)

        if loss_type == "time_l1":
            loss_tmp = torch.abs(y_pre - x_wav)
            return 0.5 * torch.sum(loss_tmp)
        elif loss_type == "time_mse":
            loss_tmp =  (y_pre - x_wav) **2
            return 0.5 * torch.sum(loss_tmp)
        elif loss_type == "spec_mae":
            # 直接在训练域复谱上做 L1（复数用 abs 表示复距离）
            loss_tmp = torch.abs(Y_pre - X)
            return 0.5 * torch.sum(loss_tmp)

        elif loss_type == "spec_mse":  #这个似乎是storm默认loss
            # 直接在训练域复谱上做 L2
            loss_tmp = torch.abs(Y_pre - X) ** 2
            return 0.5 * torch.sum(loss_tmp)
        elif loss_type == "spec_cplx_mag":
            # 复谱差 + 幅度差，通常比纯 time_mse 更像 storm 那种约束思路
            loss_tmp = torch.abs(Y_pre - X) ** 2
            loss_cplx = 0.5 * torch.sum(loss_tmp)
            loss_tmp = (torch.abs(Y_pre) - torch.abs(X)) ** 2
            loss_mag = 0.5 * torch.sum(loss_tmp)
            return loss_cplx + 0.5 * loss_mag

        elif loss_type == "none":
            return torch.zeros((), device=x_wav.device, dtype=x_wav.dtype)

        else:
            raise ValueError(f"Unsupported frontend_aux_loss: {loss_type}")

    def _compute_selector_projection_oracle_loss(
        self,
        X: torch.Tensor,
        Y_noisy: torch.Tensor,
        Y_pre: torch.Tensor,
        selector_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Projection-oracle loss for tfgrid_a selector mask.

        tfgrid_a constructs:
            Y_hint = Y_pre + M * (Y_noisy - Y_pre)

        For each time-frequency bin, compute the closed-form optimal real-valued
        interpolation coefficient M_star in [0, 1]:
            M_star = argmin_M |Y_pre + M*(Y_noisy - Y_pre) - X|^2

        Then regress the predicted selector mask M to M_star.  X/Y_noisy/Y_pre
        are detached, so this auxiliary loss updates the selector mask path only.
        """
        eps = max(float(getattr(self, "selector_oracle_eps", 1e-8)), 1e-12)
        zero = torch.zeros((), device=X.device, dtype=torch.float32)
        if selector_mask is None:
            return zero, {}

        # Predicted mask M should remain differentiable.
        M = selector_mask
        if torch.is_complex(M):
            M = M.real
        M = M.float()

        # If a previous interpolate produced spatial mismatch, align to X.
        if M.dim() != 4:
            raise RuntimeError(f"selector_mask must be 4D [B,C,F,T], got {tuple(M.shape)}")
        if M.shape[-2:] != X.shape[-2:]:
            M = F.interpolate(M, size=X.shape[-2:], mode="bilinear", align_corners=False)
        if M.size(1) != 1:
            M = M.mean(dim=1, keepdim=True)

        # Stop gradients through clean/noisy/frontend spectra; only train selector mask.
        X_det = X.detach()
        Y_noisy_det = Y_noisy.detach()
        Y_pre_det = Y_pre.detach()

        # Projection oracle: min_m |Y_pre + m*(Y_noisy - Y_pre) - X|^2.
        D = Y_noisy_det - Y_pre_det
        R = X_det - Y_pre_det

        if torch.is_complex(D):
            numerator = torch.real(R * torch.conj(D))
            denominator = torch.abs(D).pow(2).clamp_min(eps)
        else:
            numerator = R * D
            denominator = D.pow(2).clamp_min(eps)

        M_star = (numerator / denominator).clamp(0.0, 1.0).float().detach()
        if M_star.dim() == 4 and M_star.size(1) != 1:
            M_star = M_star.mean(dim=1, keepdim=True)

        # Confidence: supervise more where Y_noisy and Y_pre are meaningfully different.
        # This avoids over-constraining bins where the interpolation direction is tiny.
        if torch.is_complex(D):
            diff_energy = torch.abs(D).pow(2)
            scale_energy = torch.abs(Y_noisy_det).pow(2) + torch.abs(Y_pre_det).pow(2)
        else:
            diff_energy = D.pow(2)
            scale_energy = Y_noisy_det.pow(2) + Y_pre_det.pow(2)
        conf = (diff_energy / (scale_energy + eps)).detach().float().clamp(0.0, 1.0)
        if conf.dim() == 4 and conf.size(1) != 1:
            conf = conf.mean(dim=1, keepdim=True)

        conf_mix = min(max(float(getattr(self, "selector_oracle_conf_weight", 1.0)), 0.0), 1.0)
        weight_map = (1.0 - conf_mix) + conf_mix * conf

        # SmoothL1 is better than BCE here because M_star is a continuous coefficient.
        loss_map = F.smooth_l1_loss(M, M_star, reduction="none")
        loss = (weight_map * loss_map).mean()

        stats = {
            "selector_oracle_raw": loss.detach(),
            "selector_mask_mean": M.detach().mean(),
            "selector_mask_std": M.detach().std(),
            "selector_target_mean": M_star.detach().mean(),
            "selector_target_std": M_star.detach().std(),
            "selector_conf_mean": conf.detach().mean(),
        }
        return loss, stats
        
    def _compute_residual_regen_oracle_loss(
        self,
        X: torch.Tensor,
        Y_noisy: torch.Tensor,
        x_hat_spec: torch.Tensor,
        residual_gates,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Oracle loss for residual_regen channel gates.

        The residual_regen selector produces channel-wise gates [B,C,1,1]
        inside decoder levels.  We supervise these gates with a residual speech
        leakage ratio:

            target = |X - x_hat|^p / (|X - x_hat|^p + |Y_noisy - X|^p + eps)

        If the current estimate x_hat misses speech, X-x_hat is large, target
        is high, and the gate is encouraged to allow a stronger feature update.
        If the residual is dominated by true noise, target is low.
        """
        eps = max(float(getattr(self, "selector_oracle_eps", 1e-8)), 1e-12)
        zero = torch.zeros((), device=X.device, dtype=torch.float32)
        if x_hat_spec is None or residual_gates is None:
            return zero, {}
        if torch.is_tensor(residual_gates):
            gates = [residual_gates]
        else:
            gates = list(residual_gates)
        gates = [g for g in gates if g is not None]
        if len(gates) == 0:
            return zero, {}

        X_det = X.detach()
        Y_noisy_det = Y_noisy.detach()
        x_hat_det = x_hat_spec.detach()

        speech_leak = torch.abs(X_det - x_hat_det).pow(2)
        true_noise = torch.abs(Y_noisy_det - X_det).pow(2)
        target_map = speech_leak / (speech_leak + true_noise + eps)
        # One target per utterance; expand to each channel-wise gate.
        target_scalar = target_map.mean(dim=(1, 2, 3), keepdim=True).float().clamp(0.0, 1.0)

        losses = []
        gate_means = []
        gate_stds = []
        for gate in gates:
            g = gate.float()
            tgt = target_scalar.to(device=g.device, dtype=g.dtype).expand_as(g)
            losses.append(F.smooth_l1_loss(g, tgt, reduction="mean"))
            gate_means.append(g.detach().mean())
            gate_stds.append(g.detach().std())
        loss = torch.stack(losses).mean()

        stats = {
            "residual_regen_oracle_raw": loss.detach(),
            "residual_gate_mean": torch.stack(gate_means).mean(),
            "residual_gate_std": torch.stack(gate_stds).mean(),
            "residual_gate_target_mean": target_scalar.detach().mean(),
            "residual_gate_target_std": target_scalar.detach().std(),
        }
        return loss, stats

    def _compute_noise_prior_oracle_loss(
        self,
        X: torch.Tensor,
        Y_noisy: torch.Tensor,
        Y_pre: torch.Tensor,
        noise_prior_gates,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Oracle loss for noise_prior_guidance gates.

        noise_prior_guidance uses |Y_noisy - Y_pre| only as a noise prior, not
        as speech evidence.  Therefore a high gate should mean: the difference
        between noisy and the predictive estimate is likely dominated by true
        noise, so noise-aware modulation should be active.

        We use an utterance-level oracle ratio:

            target = |Y_noisy - X|^2 / (|Y_noisy - X|^2 + |X - Y_pre|^2 + eps)

        where |Y_noisy - X| is the true noise energy and |X - Y_pre| is the
        speech/detail error left by the frontend.  If the residual contains
        mostly noise, target is high. If the frontend has removed useful speech
        detail and Y_pre is far from X, target is lower, discouraging the
        noise-prior gate from treating the whole residual as noise.
        """
        eps = max(float(getattr(self, "selector_oracle_eps", 1e-8)), 1e-12)
        zero = torch.zeros((), device=X.device, dtype=torch.float32)
        if noise_prior_gates is None:
            return zero, {}
        if torch.is_tensor(noise_prior_gates):
            gates = [noise_prior_gates]
        else:
            gates = list(noise_prior_gates)
        gates = [g for g in gates if g is not None]
        if len(gates) == 0:
            return zero, {}

        X_det = X.detach()
        Y_noisy_det = Y_noisy.detach()
        Y_pre_det = Y_pre.detach()

        true_noise = torch.abs(Y_noisy_det - X_det).pow(2)
        frontend_error = torch.abs(X_det - Y_pre_det).pow(2)
        target_map = true_noise / (true_noise + frontend_error + eps)
        target_scalar = target_map.mean(dim=(1, 2, 3), keepdim=True).float().clamp(0.0, 1.0)

        losses = []
        gate_means = []
        gate_stds = []
        for gate in gates:
            g = gate.float()
            tgt = target_scalar.to(device=g.device, dtype=g.dtype).expand_as(g)
            losses.append(F.smooth_l1_loss(g, tgt, reduction="mean"))
            gate_means.append(g.detach().mean())
            gate_stds.append(g.detach().std())
        loss = torch.stack(losses).mean()

        stats = {
            "noise_prior_oracle_raw": loss.detach(),
            "noise_prior_gate_mean": torch.stack(gate_means).mean(),
            "noise_prior_gate_std": torch.stack(gate_stds).mean(),
            "noise_prior_gate_target_mean": target_scalar.detach().mean(),
            "noise_prior_gate_target_std": target_scalar.detach().std(),
        }
        return loss, stats


    @staticmethod
    def _complex_inner_sum(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """Per-sample real inner product for real or complex tensors."""
        if torch.is_complex(a) or torch.is_complex(b):
            prod = torch.real(a * torch.conj(b))
        else:
            prod = a * b
        return prod.flatten(start_dim=1).sum(dim=1)

    @staticmethod
    def _energy_sum(a: torch.Tensor) -> torch.Tensor:
        return torch.abs(a).pow(2).flatten(start_dim=1).sum(dim=1)

    @staticmethod
    def _band_edges(freq_bins: int, num_bands: int, device=None) -> torch.Tensor:
        nb = max(1, min(int(num_bands), int(freq_bins)))
        return torch.linspace(0, int(freq_bins), nb + 1, device=device).round().long()

    @classmethod
    def _complex_inner_band_sum(cls, a: torch.Tensor, b: torch.Tensor, num_bands: int) -> torch.Tensor:
        """Per-sample, per-band real inner product. Return [B,1,N_band,1]."""
        if torch.is_complex(a) or torch.is_complex(b):
            prod = torch.real(a * torch.conj(b))
        else:
            prod = a * b
        B = prod.shape[0]
        freq_bins = int(prod.shape[-2])
        edges = cls._band_edges(freq_bins, num_bands, device=prod.device)
        vals = []
        for i in range(edges.numel() - 1):
            lo = int(edges[i].item())
            hi = int(edges[i + 1].item())
            if hi <= lo:
                hi = min(lo + 1, freq_bins)
            vals.append(prod[..., lo:hi, :].reshape(B, -1).sum(dim=1))
        return torch.stack(vals, dim=1).view(B, 1, -1, 1)

    @classmethod
    def _energy_band_sum(cls, a: torch.Tensor, num_bands: int) -> torch.Tensor:
        """Per-sample, per-band energy. Return [B,1,N_band,1]."""
        mag2 = torch.abs(a).pow(2)
        B = mag2.shape[0]
        freq_bins = int(mag2.shape[-2])
        edges = cls._band_edges(freq_bins, num_bands, device=mag2.device)
        vals = []
        for i in range(edges.numel() - 1):
            lo = int(edges[i].item())
            hi = int(edges[i + 1].item())
            if hi <= lo:
                hi = min(lo + 1, freq_bins)
            vals.append(mag2[..., lo:hi, :].reshape(B, -1).sum(dim=1))
        return torch.stack(vals, dim=1).view(B, 1, -1, 1)

    @staticmethod
    def _expand_band_values(x_band: torch.Tensor, freq_bins: int) -> torch.Tensor:
        """Nearest-expand [B,1,N_band,1] to [B,1,F,1]."""
        if int(x_band.shape[-2]) == int(freq_bins):
            return x_band
        return F.interpolate(x_band, size=(int(freq_bins), 1), mode="nearest")

    def _main_and_cond_to_pre_noisy(self, y_main: torch.Tensor, y_cond: Optional[torch.Tensor]):
        """Map sampler/score inputs to (Y_pre, Y_noisy) according to train_type."""
        if y_cond is None:
            return y_main, y_main
        if self.train_type == "mode2":
            # mode2: SDE center is frontend enhanced spectrum; y_cond is original noisy.
            return y_main, y_cond
        else:
            # mode3: y_main is noisy; y_cond is frontend enhanced.
            return y_cond, y_main

    def _ouve_score_to_x0(
        self,
        x_t: torch.Tensor,
        y_main: torch.Tensor,
        t: torch.Tensor,
        score: torch.Tensor,
    ) -> torch.Tensor:
        """Convert an OUVE score estimate to the corresponding clean x0 estimate."""
        if self.sde.__class__.__name__ != "OUVESDE":
            raise ValueError("Anchor V2 currently supports OUVESDE only.")
        if t.dim() == 0:
            t = torch.full((x_t.shape[0],), float(t.item()), device=x_t.device, dtype=torch.float32)
        else:
            t = t.to(device=x_t.device, dtype=torch.float32).flatten()
        alpha = self.sde.alpha(t).view(-1, 1, 1, 1).to(dtype=x_t.real.dtype if torch.is_complex(x_t) else x_t.dtype)
        sigma = self.sde._std(t).view(-1, 1, 1, 1).to(dtype=alpha.dtype)
        mean_hat = x_t + sigma.square() * score
        return (mean_hat - (1.0 - alpha) * y_main) / alpha.clamp_min(1e-5)

    def build_anchor_v2_score_calibrator(
        self,
        y_main: torch.Tensor,
        y_cond: Optional[torch.Tensor] = None,
        a_max_override: Optional[float] = None,
        b_max_override: Optional[float] = None,
        scale: Optional[float] = None,
        single_shot: bool = False,
        single_shot_ratio: float = 0.5,
        single_shot_step: Optional[int] = None,
    ):
        """Return a predictor-score calibrator for fine-grained Anchor V2.

        Unlike the original anchor, V2 does not shift the already-discretized
        transition mean.  It calibrates the score network's x0 estimate inside
        the two-dimensional conditional evidence subspace and converts that
        residual back to a score residual before the predictor integrates it.
        """
        if self.anchor_v2_controller is None:
            return None
        if self.sde.__class__.__name__ != "OUVESDE":
            warnings.warn("anchor_v2 is currently implemented for OUVESDE only; disabling it.")
            return None

        Y_pre, Y_noisy = self._main_and_cond_to_pre_noisy(y_main, y_cond)
        t_max = float(getattr(self.sde, "T", 1.0))
        single_shot_ratio = float(single_shot_ratio)
        if not 0.0 <= single_shot_ratio <= 1.0:
            raise ValueError("anchor_v2_single_shot_ratio must be in [0,1].")
        if single_shot_step is not None and int(single_shot_step) < 1:
            raise ValueError("anchor_v2_single_shot_step is 1-based and must be >= 1.")

        def _score_calibrator(
            x_t,
            y_unused,
            t,
            score_base,
            step_index=None,
            num_steps=None,
        ):
            if bool(single_shot):
                if step_index is None or num_steps is None:
                    raise RuntimeError("Single-shot Anchor V2 requires sampler step metadata.")
                n = max(int(num_steps), 1)
                if single_shot_step is not None:
                    target_step = min(int(single_shot_step) - 1, n - 1)
                else:
                    target_step = int(round(single_shot_ratio * max(n - 1, 0)))
                if int(step_index) != target_step:
                    return score_base

            x0_base = self._ouve_score_to_x0(x_t, y_main, t, score_base)
            out = self.anchor_v2_controller(
                x_t=x_t,
                x0_base=x0_base,
                y_pre=Y_pre,
                y_noisy=Y_noisy,
                t=t,
                t_max=t_max,
                detach_inputs=True,
                a_max_override=a_max_override,
                b_max_override=b_max_override,
                scale=scale,
            )
            if t.dim() == 0:
                t_vec = torch.full((x_t.shape[0],), float(t.item()), device=x_t.device, dtype=torch.float32)
            else:
                t_vec = t.to(device=x_t.device, dtype=torch.float32).flatten()
            alpha = self.sde.alpha(t_vec).view(-1, 1, 1, 1)
            sigma2 = self.sde._std(t_vec).view(-1, 1, 1, 1).square().clamp_min(1e-8)
            delta_score = alpha.to(dtype=out["delta_x0"].real.dtype if torch.is_complex(out["delta_x0"]) else out["delta_x0"].dtype) \
                * out["delta_x0"] / sigma2.to(dtype=alpha.dtype)
            score_calib = score_base + delta_score.to(dtype=score_base.dtype)
            if torch.is_complex(score_calib):
                score_calib = torch.complex(
                    torch.nan_to_num(score_calib.real),
                    torch.nan_to_num(score_calib.imag),
                )
            else:
                score_calib = torch.nan_to_num(score_calib)
            return score_calib

        return _score_calibrator

    def _compute_anchor_v2_loss(
        self,
        X: torch.Tensor,
        Y_noisy: torch.Tensor,
        Y_pre: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Train Anchor V2 against the score network's remaining x0 error.

        The clean target is used only in x0 coordinates.  Local oracle gates
        solve a two-direction projection of ``X-x0_base`` onto
        ``Y_pre-x0_base`` and ``Y_noisy-Y_pre`` in each frequency-band/time
        tile.  Gradients are restricted to the V2 controller.
        """
        zero = torch.zeros((), device=X.device, dtype=torch.float32)
        if self.anchor_v2_controller is None or float(getattr(self, "anchor_v2_weight", 0.0)) <= 0.0:
            return zero, {}
        if self.sde.__class__.__name__ != "OUVESDE":
            return zero, {}

        B = X.shape[0]
        device = X.device
        t_eps = float(getattr(self, "t_eps", 0.03))
        T = float(getattr(self.sde, "T", 1.0))
        if bool(getattr(self, "anchor_v2_single_shot_train", False)):
            center = float(getattr(self, "anchor_v2_single_shot_ratio", 0.5))
            width = float(getattr(self, "anchor_v2_single_shot_train_width", 0.10))
            progress_min = max(0.0, center - width)
            progress_max = min(1.0, center + width)
            reverse_progress = progress_min + torch.rand(B, device=device) * (progress_max - progress_min)
            # Reverse progress 0 corresponds to t=T; progress 1 corresponds to t=t_eps.
            t = T - reverse_progress * (T - t_eps)
        else:
            t = torch.rand(B, device=device) * (T - t_eps) + t_eps

        if self.train_type == "mode2":
            y_main, y_cond = Y_pre, Y_noisy
        else:
            y_main, y_cond = Y_noisy, Y_pre

        mean, std = self.sde.marginal_prob(X, y_main, t)
        x_t = mean + std[:, None, None, None] * torch.randn_like(X)

        # Anchor V2 learns a residual on top of a frozen score estimate.
        with torch.no_grad():
            score_base = self(x_t, y_main, t, y_cond=y_cond)
            x0_base = self._ouve_score_to_x0(x_t, y_main, t, score_base).detach()

        pred = self.anchor_v2_controller(
            x_t=x_t,
            x0_base=x0_base,
            y_pre=Y_pre,
            y_noisy=Y_noisy,
            t=t,
            t_max=T,
            detach_inputs=True,
        )

        residual = X.detach() - x0_base
        u_pre = Y_pre.detach() - x0_base
        u_noisy = Y_noisy.detach() - Y_pre.detach()
        a_star_raw, b_star_raw, oracle_stats = solve_two_direction_tile_oracle(
            residual=residual,
            u_pre=u_pre,
            u_noisy=u_noisy,
            num_bands=int(self.anchor_v2_controller.num_bands),
            num_time_blocks=int(self.anchor_v2_controller.num_time_blocks),
            eps=float(getattr(self, "anchor_v2_oracle_eps", 1e-6)),
        )

        a_cap = pred["a_cap"].expand_as(a_star_raw).detach()
        b_cap = pred["b_cap"].expand_as(b_star_raw).detach()
        a_star = torch.minimum(torch.maximum(a_star_raw, -a_cap), a_cap).detach()
        b_star = torch.minimum(torch.maximum(b_star_raw, -b_cap), b_cap).detach()

        # Downweight nearly collinear/silent tiles while retaining a small
        # learning signal so the controller does not ignore difficult regions.
        oracle_weight = oracle_stats["conditioning"].detach().clamp_min(0.05)

        a_den = a_cap.detach().clamp_min(1e-3)
        b_den = b_cap.detach().clamp_min(1e-3)

        loss_a_map = F.smooth_l1_loss(
            pred["a_tile"].float() / a_den.float(),
            a_star.float() / a_den.float(),
            reduction="none",
        )

        loss_b_map = F.smooth_l1_loss(
            pred["b_tile"].float() / b_den.float(),
            b_star.float() / b_den.float(),
            reduction="none",
        )

        loss_a = (loss_a_map * oracle_weight).sum() / oracle_weight.sum().clamp_min(1.0)
        loss_b = (loss_b_map * oracle_weight).sum() / oracle_weight.sum().clamp_min(1.0)
        
        target_real = torch.view_as_real(X.detach()) if torch.is_complex(X) else X.detach()
        calib_real = torch.view_as_real(pred["x0_calib"]) if torch.is_complex(pred["x0_calib"]) else pred["x0_calib"]
        loss_recon = F.smooth_l1_loss(calib_real, target_real, reduction="mean")

        reduce_dims = tuple(range(1, X.dim()))
        base_err = torch.abs(x0_base - X.detach()).float().square().mean(dim=reduce_dims)
        calib_err = torch.abs(pred["x0_calib"] - X.detach()).float().square().mean(dim=reduce_dims)
        rel_degradation = (calib_err - base_err) / base_err.detach().clamp_min(1e-8)
        loss_gain = F.relu(rel_degradation).mean()

        loss_smooth = torch.zeros((), device=device, dtype=torch.float32)
        for gate in (pred["a_tile"].float(), pred["b_tile"].float()):
            if gate.shape[-2] > 1:
                loss_smooth = loss_smooth + gate.diff(dim=-2).abs().mean()
            if gate.shape[-1] > 1:
                loss_smooth = loss_smooth + gate.diff(dim=-1).abs().mean()
        loss_sparse = pred["a_tile"].float().abs().mean() + pred["b_tile"].float().abs().mean()

        loss = (
            float(getattr(self, "anchor_v2_loss_weight_a", 1.0)) * loss_a
            + float(getattr(self, "anchor_v2_loss_weight_b", 1.0)) * loss_b
            + float(getattr(self, "anchor_v2_loss_weight_recon", 0.1)) * loss_recon
            + float(getattr(self, "anchor_v2_loss_weight_gain", 0.1)) * loss_gain
            + float(getattr(self, "anchor_v2_smooth_weight", 0.01)) * loss_smooth
            + float(getattr(self, "anchor_v2_sparsity_weight", 0.001)) * loss_sparse
        )

        eps = 1e-8
        delta_ratio = (
            torch.abs(pred["delta_x0"]).float().square().mean().sqrt()
            / torch.abs(x0_base).float().square().mean().sqrt().clamp_min(eps)
        )
        stats = {
            "anchor_v2_loss_a": loss_a.detach(),
            "anchor_v2_loss_b": loss_b.detach(),
            "anchor_v2_loss_recon": loss_recon.detach(),
            "anchor_v2_loss_gain": loss_gain.detach(),
            "anchor_v2_loss_smooth": loss_smooth.detach(),
            "anchor_v2_loss_sparse": loss_sparse.detach(),
            "anchor_v2_a_abs_mean": pred["a_tile"].detach().float().abs().mean(),
            "anchor_v2_b_abs_mean": pred["b_tile"].detach().float().abs().mean(),
            "anchor_v2_a_target_abs_mean": a_star.detach().float().abs().mean(),
            "anchor_v2_b_target_abs_mean": b_star.detach().float().abs().mean(),
            "anchor_v2_oracle_conditioning": oracle_stats["conditioning"].detach().mean(),
            "anchor_v2_improve_fraction": (calib_err < base_err).detach().float().mean(),
            "anchor_v2_relative_error_change": rel_degradation.detach().mean(),
            "anchor_v2_delta_ratio": delta_ratio.detach(),
            "anchor_v2_E_mean": pred["E_t"].detach().mean(),
            "anchor_v2_t_mean": t.detach().float().mean(),
            "anchor_v2_reverse_progress_mean": ((T - t.detach().float()) / max(T - t_eps, 1e-8)).mean(),
        }
        return loss, stats

    def build_anchor_calibrator(
        self,
        y_main: torch.Tensor,
        y_cond: Optional[torch.Tensor] = None,
        k_max_override: Optional[float] = None,
        k_scale: Optional[float] = None,
    ):
        """Return a callable used by the PC sampler to calibrate predictor mean.

        The callable receives (x_t, mu_base, y_main, t_next) and returns a calibrated
        mean.  If the controller is disabled, returns None.

        k_max_override and k_scale are inference-time controls.  They do not
        modify checkpoint weights; they only change the pull strength used by
        the sampler:
            K_t = (k_max_override or checkpoint K_max) * k_scale * E(t) * g_t.
        """
        if self.anchor_controller is None:
            return None
        Y_pre, Y_noisy = self._main_and_cond_to_pre_noisy(y_main, y_cond)
        t_max = float(getattr(self.sde, "T", 1.0))

        def _calibrator(x_t, mu_base, y_unused, t_next):
            out = self.anchor_controller(
                x_t=x_t,
                mu_base=mu_base,
                y_pre=Y_pre,
                y_noisy=Y_noisy,
                t=t_next,
                t_max=t_max,
                detach_inputs=True,
                k_max_override=k_max_override,
                k_scale=k_scale,
            )
            return out["mu_anchor"]

        return _calibrator

    def _compute_anchor_calib_loss(
        self,
        X: torch.Tensor,
        Y_noisy: torch.Tensor,
        Y_pre: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Train the state-aware anchor controller with clean-referenced oracles.

        The score network is first used to obtain the base OUVE reverse mean
        mu_base at a random time t.  The controller then predicts:
            c_t: A_t = c_t Y_pre + (1-c_t)Y_noisy
            K_t: final pull strength

        We supervise c_t with the optimal projection coefficient between
        Y_noisy and Y_pre, and supervise K_t with the optimal projection from
        mu_base toward the predicted anchor direction.  This trains the
        controller to answer: "does moving the base transition mean toward this
        conditional anchor reduce clean-reference error?".
        """
        zero = torch.zeros((), device=X.device, dtype=torch.float32)
        if self.anchor_controller is None or float(getattr(self, "anchor_calib_weight", 0.0)) <= 0.0:
            return zero, {}
        if self.sde.__class__.__name__ != "OUVESDE":
            return zero, {}

        B = X.shape[0]
        device = X.device
        eps = max(float(getattr(self, "selector_oracle_eps", 1e-8)), 1e-12)
        t_eps = float(getattr(self, "t_eps", 0.03))
        T = float(getattr(self.sde, "T", 1.0))
        t = torch.rand(B, device=device) * (T - t_eps) + t_eps

        if self.train_type == "mode2":
            y_main = Y_pre
            y_cond = Y_noisy
        else:
            y_main = Y_noisy
            y_cond = Y_pre

        mean, std = self.sde.marginal_prob(X, y_main, t)
        z = torch.randn_like(X)
        x_t = mean + std[:, None, None, None] * z

        # Obtain base OUVE reverse mean without updating the score network through this auxiliary loss.
        with torch.no_grad():
            def score_fn(xx, yy, tt):
                return self(xx, yy, tt, y_cond=y_cond)
            rsde = self.sde.reverse(score_fn, probability_flow=False)
            stepsize = torch.tensor(1.0 / max(int(getattr(self.sde, "N", 30)), 1), device=device, dtype=t.dtype)
            f, _ = rsde.discretize(x_t, y_main, t, stepsize)
            mu_base = (x_t - f).detach()

        # Controller prediction keeps gradients only through controller parameters.
        pred = self.anchor_controller(
            x_t=x_t,
            mu_base=mu_base,
            y_pre=Y_pre,
            y_noisy=Y_noisy,
            t=t,
            t_max=T,
            detach_inputs=True,
        )
        c_t = pred["c_t"].float()
        K_t = pred["K_t"].float()
        A_t = pred["A_t"]
        mu_anchor = pred["mu_anchor"]
        E_t = pred["E_t"].float()

        is_band = getattr(self.anchor_controller, "controller_type", "global_mlp") == "band_mlp"

        # Oracle c*: closest point on the segment Y_noisy -> Y_pre to clean X.
        D_cond = (Y_pre.detach() - Y_noisy.detach())
        R_cond = (X.detach() - Y_noisy.detach())

        # Oracle K*: closest projection from mu_base toward the predicted anchor A_t.
        D_anchor = (A_t.detach() - mu_base.detach())
        R_anchor = (X.detach() - mu_base.detach())

        if is_band:
            num_bands = int(getattr(self.anchor_controller, "num_bands", 32))
            freq_bins = int(X.shape[-2])
            # Band-wise oracle targets.  Loss is computed on band outputs to avoid
            # over-weighting wider expanded bands.
            c_star_band = (
                self._complex_inner_band_sum(R_cond, D_cond, num_bands)
                / (self._energy_band_sum(D_cond, num_bands) + eps)
            ).clamp(0.0, 1.0).float().detach()
            c_star = self._expand_band_values(c_star_band, freq_bins=freq_bins).detach()

            K_raw_band = self._complex_inner_band_sum(R_anchor, D_anchor, num_bands) / (self._energy_band_sum(D_anchor, num_bands) + eps)
            K_cap_band = (float(self.anchor_controller.k_max) * E_t).expand_as(K_raw_band).clamp_min(0.0).detach()
            K_star_band = torch.minimum(torch.maximum(K_raw_band, torch.zeros_like(K_raw_band)), K_cap_band).float().detach()
            K_star = self._expand_band_values(K_star_band, freq_bins=freq_bins).detach()

            c_pred_for_loss = pred.get("c_band", c_t).float()
            K_pred_for_loss = pred.get("K_band", K_t).float()
            loss_c = F.smooth_l1_loss(c_pred_for_loss, c_star_band, reduction="mean")
            loss_k = F.smooth_l1_loss(K_pred_for_loss, K_star_band, reduction="mean")

            loss_smooth = torch.zeros((), device=X.device, dtype=torch.float32)
            smooth_w = float(getattr(self, "anchor_band_smooth_weight", 0.0))
            if smooth_w > 0.0:
                c_band = pred.get("c_band", None)
                g_band = pred.get("g_band", None)
                if c_band is not None and g_band is not None and c_band.shape[-2] > 1:
                    loss_smooth = (
                        c_band.float().diff(dim=-2).abs().mean()
                        + g_band.float().diff(dim=-2).abs().mean()
                    )
        else:
            c_star = (self._complex_inner_sum(R_cond, D_cond) / (self._energy_sum(D_cond) + eps)).clamp(0.0, 1.0)
            c_star = c_star.view(B, 1, 1, 1).float().detach()

            K_raw = self._complex_inner_sum(R_anchor, D_anchor) / (self._energy_sum(D_anchor) + eps)
            K_cap = (float(self.anchor_controller.k_max) * E_t.view(B)).clamp_min(0.0).detach()
            K_star = torch.minimum(torch.maximum(K_raw, torch.zeros_like(K_raw)), K_cap).view(B, 1, 1, 1).float().detach()

            loss_c = F.smooth_l1_loss(c_t, c_star, reduction="mean")
            loss_k = F.smooth_l1_loss(K_t, K_star, reduction="mean")
            loss_smooth = torch.zeros((), device=X.device, dtype=torch.float32)

        loss_recon = F.smooth_l1_loss(torch.view_as_real(mu_anchor) if torch.is_complex(mu_anchor) else mu_anchor,
                                      torch.view_as_real(X.detach()) if torch.is_complex(X) else X.detach(),
                                      reduction="mean")
        loss = (
            float(getattr(self, "anchor_loss_weight_c", 1.0)) * loss_c
            + float(getattr(self, "anchor_loss_weight_k", 1.0)) * loss_k
            + float(getattr(self, "anchor_loss_weight_recon", 0.1)) * loss_recon
            + float(getattr(self, "anchor_band_smooth_weight", 0.0)) * loss_smooth
        )
        stats = {
            "anchor_loss_c": loss_c.detach(),
            "anchor_loss_k": loss_k.detach(),
            "anchor_loss_recon": loss_recon.detach(),
            "anchor_loss_smooth": loss_smooth.detach(),
            "anchor_c_mean": c_t.detach().mean(),
            "anchor_c_target_mean": c_star.detach().mean(),
            "anchor_K_mean": K_t.detach().mean(),
            "anchor_K_target_mean": K_star.detach().mean(),
            "anchor_g_mean": pred["g_t"].detach().mean(),
            "anchor_E_mean": E_t.detach().mean(),
        }
        return loss, stats


    def _prepare_wave_for_content(self, wav: torch.Tensor) -> torch.Tensor:
        # [B,1,T] / [B,T] / [T] -> [B,T]
        if wav.dim() == 3 and wav.size(1) == 1:
            wav = wav[:, 0, :]
        elif wav.dim() == 1:
            wav = wav.unsqueeze(0)
        elif wav.dim() != 2:
            raise RuntimeError(f"_prepare_wave_for_content expects [T], [B,T] or [B,1,T], got {tuple(wav.shape)}")

        wav = wav.float()
        wav = wav - wav.mean(dim=-1, keepdim=True)
        wav = wav / wav.abs().amax(dim=-1, keepdim=True).clamp_min(1e-6)

        if int(self.sr) != 16000:
            wav = AF.resample(wav, orig_freq=int(self.sr), new_freq=16000)

        return wav

    def _extract_content_hidden(self, wav: torch.Tensor, detach_target: bool = False) -> torch.Tensor:
        if self.content_model is None:
            raise RuntimeError("content_model is None but _extract_content_hidden was called.")

        wav = self._prepare_wave_for_content(wav)

        if detach_target:
            with torch.no_grad():
                out = self.content_model(
                    input_values=wav,
                    output_hidden_states=True,
                    return_dict=True,
                )
        else:
            out = self.content_model(
                input_values=wav,
                output_hidden_states=True,
                return_dict=True,
            )

        hs = out.hidden_states
        if hs is None:
            feat = out.last_hidden_state
        else:
            idx = self.content_layer
            if idx < 0:
                idx = len(hs) + idx
            idx = max(0, min(idx, len(hs) - 1))
            feat = hs[idx]

        return feat

    def _content_regression_loss_from_wavs(self, pred_wav: torch.Tensor, target_wav: torch.Tensor) -> torch.Tensor:
        if (not self.use_content_loss) or (self.content_model is None):
            return torch.zeros((), device=pred_wav.device, dtype=pred_wav.dtype)

        feat_pred = self._extract_content_hidden(pred_wav, detach_target=False)
        feat_tgt = self._extract_content_hidden(target_wav, detach_target=True)

        if feat_pred.shape[1] != feat_tgt.shape[1]:
            feat_pred = F.interpolate(
                feat_pred.transpose(1, 2),
                size=feat_tgt.shape[1],
                mode="linear",
                align_corners=False,
            ).transpose(1, 2)

        feat_pred = F.layer_norm(feat_pred, (feat_pred.shape[-1],))
        feat_tgt = F.layer_norm(feat_tgt, (feat_tgt.shape[-1],))

        if self.content_loss_type == "l1":
            return F.l1_loss(feat_pred, feat_tgt)
        elif self.content_loss_type == "mse":
            return F.mse_loss(feat_pred, feat_tgt)
        elif self.content_loss_type == "smooth_l1":
            return F.smooth_l1_loss(feat_pred, feat_tgt)
        elif self.content_loss_type == "cosine":
            return 1.0 - F.cosine_similarity(feat_pred, feat_tgt, dim=-1).mean()
        else:
            raise ValueError(f"Unsupported content_loss_type: {self.content_loss_type}")
    
    def _build_predictive_residual_state(
        self,
        X: torch.Tensor,
        Y_noisy: torch.Tensor,
        Y_pre: torch.Tensor,
    ):
        """Build residual-coordinate state for training/validation.

        R_target = X - Y_pre is the clean correction that the backend should
        recover. R_obs = Y_noisy - Y_pre = R_target + N is available at
        inference. The configured residual-prior network returns either the
        predictive complex residual R_theta or G * R_obs (confidence gate).
        OUVE applies lambda(t) to this prior internally, so r_center itself
        remains the unscaled prediction and can be shared by training/sampling.
        """
        if not self.residual_refine_enable or self.residual_initializer is None:
            raise RuntimeError("Predictive residual refinement is not enabled.")
        y_pre_det = Y_pre.detach()
        y_noisy_det = Y_noisy.detach()
        r_target = X.detach() - y_pre_det
        r_obs = y_noisy_det - y_pre_det
        r_prior = self.residual_initializer(y_pre_det, y_noisy_det)
        r_center = r_prior if self.residual_init_joint_grad else r_prior.detach()
        return r_target, r_obs, r_prior, r_center

    @staticmethod
    def _residual_snr_db(target: torch.Tensor, estimate: torch.Tensor, eps: float = 1e-8):
        target_energy = torch.abs(target).pow(2).flatten(start_dim=1).sum(dim=1)
        error_energy = torch.abs(estimate - target).pow(2).flatten(start_dim=1).sum(dim=1)
        return 10.0 * torch.log10((target_energy + eps) / (error_energy + eps))

    def _compute_residual_initializer_loss(
        self,
        r_target: torch.Tensor,
        r_obs: torch.Tensor,
        r_det: torch.Tensor,
    ):
        """Supervise the residual prior and report collapse-sensitive metrics.

        For confidence_gate, ``r_det`` is actually R_prior = G * R_obs.  The
        main reconstruction term is supplemented by an energy-scaled oracle
        gate target.  The oracle is the local least-squares real gain and is
        used only during training/validation; inference needs no clean target.
        """
        r_prior = r_det
        loss_recon = 0.5 * torch.sum(torch.abs(r_prior - r_target).pow(2))
        loss_gate = torch.zeros((), device=r_target.device, dtype=torch.float32)
        oracle_gate = None
        gate = getattr(self.residual_initializer, "last_gate", None)

        if self.residual_prior_type == "confidence_gate":
            if gate is None:
                raise RuntimeError("confidence_gate prior did not expose last_gate")
            with torch.no_grad():
                oracle_gate = ConfidenceGatedResidualPriorPredictor.oracle_gate(
                    r_target=r_target,
                    r_obs=r_obs,
                    patch_freq=self.residual_gate_patch_freq,
                    patch_time=self.residual_gate_patch_time,
                    floor_ratio=self.residual_gate_floor_ratio,
                )
            # Weight gate mistakes by observable-residual energy. This gives the
            # auxiliary term the same physical units/scale as spectral MSE and
            # deemphasizes meaningless gates in near-silent bins.
            obs_power = torch.abs(r_obs).pow(2).float().sum(dim=1, keepdim=True).detach()
            loss_gate = 0.5 * torch.sum(obs_power * (gate.float() - oracle_gate).pow(2))

        loss = loss_recon + self.residual_gate_oracle_weight * loss_gate

        with torch.no_grad():
            snr_obs = self._residual_snr_db(r_target, r_obs)
            snr_prior = self._residual_snr_db(r_target, r_prior)
            target_energy = torch.abs(r_target).pow(2).flatten(start_dim=1).sum(dim=1)
            prior_error = torch.abs(r_prior - r_target).pow(2).flatten(start_dim=1).sum(dim=1)
            obs_error = torch.abs(r_obs - r_target).pow(2).flatten(start_dim=1).sum(dim=1)
            prior_energy = torch.abs(r_prior).pow(2).flatten(start_dim=1).sum(dim=1)
            target_rms = torch.sqrt(target_energy.mean() / max(r_target[0].numel(), 1))
            prior_rms = torch.sqrt(prior_energy.mean() / max(r_prior[0].numel(), 1))
            stats = {
                # New names.
                "residual_prior_snr_db": snr_prior.mean(),
                "residual_prior_snr_gain_over_obs_db": (snr_prior - snr_obs).mean(),
                "residual_prior_nmse": (prior_error / (target_energy + 1e-8)).mean(),
                "residual_prior_rms": prior_rms,
                "residual_prior_to_target_rms_ratio": prior_rms / (target_rms + 1e-8),
                "residual_observed_snr_db": snr_obs.mean(),
                "residual_observed_nmse": (obs_error / (target_energy + 1e-8)).mean(),
                "residual_target_rms": target_rms,
                "residual_prior_recon_loss": loss_recon.detach(),
                "residual_gate_oracle_loss": loss_gate.detach(),
                # Backward-compatible dashboard aliases.
                "residual_init_snr_db": snr_prior.mean(),
                "residual_init_snr_gain_db": (snr_prior - snr_obs).mean(),
                "residual_init_nmse": (prior_error / (target_energy + 1e-8)).mean(),
                "residual_init_rms": prior_rms,
            }
            if gate is not None and oracle_gate is not None:
                oracle_prior = r_obs * oracle_gate.to(
                    dtype=r_obs.real.dtype if torch.is_complex(r_obs) else r_obs.dtype
                )
                snr_oracle = self._residual_snr_db(r_target, oracle_prior)
                oracle_error = torch.abs(oracle_prior - r_target).pow(2).flatten(start_dim=1).sum(dim=1)
                gate_f = gate.float()
                stats.update({
                    "residual_gate_mean": gate_f.mean(),
                    "residual_gate_std": gate_f.std(unbiased=False),
                    "residual_gate_active_fraction": (gate_f > 0.10).float().mean(),
                    "residual_gate_oracle_mean": oracle_gate.mean(),
                    "residual_gate_oracle_std": oracle_gate.std(unbiased=False),
                    "residual_gate_mae": torch.abs(gate_f - oracle_gate).mean(),
                    "residual_gate_mse": torch.square(gate_f - oracle_gate).mean(),
                    "residual_oracle_prior_snr_db": snr_oracle.mean(),
                    "residual_oracle_prior_nmse": (oracle_error / (target_energy + 1e-8)).mean(),
                    "residual_prior_oracle_snr_gap_db": (snr_oracle - snr_prior).mean(),
                })
        return loss, stats

    def enhance(
        self,
        y,
        y_cond=None,
        sampler_type="pc",
        predictor="reverse_diffusion",
        corrector="none",
        N=30,
        corrector_steps=1,
        snr=0.5,
        timeit=False,
        **kwargs,
    ):
        """Use residual-anchored OUVE when enabled, otherwise base enhancement.

        In mode2 the public API stays unchanged: y is the frontend waveform and
        y_cond is the original noisy waveform. Internally the reverse process
        estimates a correction residual and returns Y_pre + R_hat.
        """
        if not self.residual_refine_enable:
            return super().enhance(
                y,
                y_cond=y_cond,
                sampler_type=sampler_type,
                predictor=predictor,
                corrector=corrector,
                N=N,
                corrector_steps=corrector_steps,
                snr=snr,
                timeit=timeit,
                **kwargs,
            )

        if y_cond is None:
            raise ValueError("Residual refinement requires both frontend and noisy waveforms.")
        if self.train_type != "mode2":
            raise ValueError("Residual refinement inference currently supports mode2 only.")

        # Standard inference entry points may always forward Anchor options.
        # Consume all of them here: residual refinement is an independent
        # ablation and must not leak unknown kwargs into the ODE solver.
        anchor_calib_arg = bool(kwargs.pop("anchor_calib", False))
        anchor_v2_arg = bool(kwargs.pop("anchor_v2", False))
        rrsd_arg = bool(kwargs.pop("rrsd", False))
        kwargs.pop("rrsd_scale", None)
        for key in (
            "anchor_k_max",
            "anchor_scale",
            "anchor_v2_a_max",
            "anchor_v2_b_max",
            "anchor_v2_scale",
            "anchor_v2_single_shot",
            "anchor_v2_single_shot_ratio",
            "anchor_v2_single_shot_step",
            "anchor_v2_branch",
        ):
            kwargs.pop(key, None)
        if anchor_calib_arg or anchor_v2_arg:
            raise ValueError("Anchor V1/V2 must be disabled for residual-refinement inference.")
        if rrsd_arg:
            raise ValueError(
                "RRSD V1-clean does not support residual-refinement inference. "
                "Use the full-speech mode2/mode3 path."
            )

        regen_strength_arg = kwargs.pop("regen_strength", None)
        if regen_strength_arg is None:
            regen_strength_arg = self.residual_refine_start_strength

        start = time.time()
        if y.dim() == 1:
            y = y[None, :]
        if y_cond.dim() == 1:
            y_cond = y_cond[None, :]
        y = y.to(self.device)
        y_cond = y_cond.to(self.device)
        T_orig = int(y_cond.size(-1))

        if y.size(-1) != T_orig:
            if y.size(-1) > T_orig:
                offset = (y.size(-1) - T_orig) // 2
                y = y[..., offset:offset + T_orig]
            else:
                pad = T_orig - y.size(-1)
                y = F.pad(y, (pad // 2, pad - pad // 2))

        # mode2 public convention: y=Y_pre waveform, y_cond=Y_noisy waveform.
        norm_factor = y_cond.abs().max().clamp_min(1e-12).item()
        y_pre_n = y / norm_factor
        y_noisy_n = y_cond / norm_factor
        Y_pre = torch.unsqueeze(self._forward_transform(self._stft(y_pre_n)), 0)
        Y_noisy = torch.unsqueeze(self._forward_transform(self._stft(y_noisy_n)), 0)
        Y_pre = pad_spec(Y_pre)
        Y_noisy = pad_spec(Y_noisy)

        r_obs = Y_noisy - Y_pre
        r_det = self.residual_initializer(Y_pre, Y_noisy)  # R_theta; OUVE applies lambda(t) internally
        score_fn = partial(self, y_cond=r_obs)
        regeneration_strength = self._resolve_regeneration_strength(
            regen_strength_arg, r_det, r_obs
        )
        if self.sde.__class__.__name__ != "OUVESDE":
            raise ValueError("Residual refinement inference currently requires OUVESDE.")
        if self.sde.sampler_type == "pc":
            sampler = self.get_pc_sampler(
                predictor,
                corrector,
                r_det,
                score_fn=score_fn,
                N=N,
                corrector_steps=corrector_steps,
                snr=snr,
                intermediate=False,
                regeneration_strength=regeneration_strength,
                **kwargs,
            )
        elif self.sde.sampler_type == "ode":
            sampler = self.get_ode_sampler(
                r_det,
                score_fn=score_fn,
                N=N,
                regeneration_strength=regeneration_strength,
                **kwargs,
            )
        else:
            raise ValueError(f"Unsupported sampler type: {self.sde.sampler_type}")

        r_hat, nfe = sampler()
        x_hat_spec = Y_pre + r_hat
        x_hat = self.to_audio(x_hat_spec.squeeze(), T_orig) * norm_factor
        x_hat = x_hat.squeeze().detach().cpu().numpy()
        if timeit:
            elapsed = time.time() - start
            rtf = elapsed / max(len(x_hat) / float(self.sr), 1e-8)
            return x_hat, nfe, rtf
        return x_hat

    def _score_step_geometry(self, step_out: dict):
        """Return OUVE score geometry shared by regret-aware objectives.

        For conditional OUVE, the perturbation mean is

            mu_t(x0, y) = w_x(t) * x0 + w_y(t) * y,

        and a score estimate implies ``mu_hat = x_t + sigma_t^2 * score``.
        Therefore the corresponding x0 estimate is obtained by inverting the
        OUVE mean, rather than by treating ``mu_hat`` itself as clean speech.

        ``sde.weights(t)`` is used instead of hard-coding ``exp(-theta*t)`` so
        the same code also remains correct for the residual-anchor schedule.
        """
        if self.sde.__class__.__name__ != "OUVESDE":
            raise RuntimeError("frontend-relative score geometry currently supports OUVESDE only.")

        score = step_out["forward_out"]
        x_t = step_out["x_t"]
        y_main = step_out["y"]
        t = step_out["t"].detach().to(device=score.device, dtype=torch.float32).flatten()

        sigma_override = step_out.get("sigma_override", None)
        if sigma_override is None:
            sigma = self.sde._std(t).to(device=score.device, dtype=score.real.dtype)
            sigma = sigma[:, None, None, None]
        else:
            sigma = sigma_override.detach().to(device=score.device, dtype=score.real.dtype)
            while sigma.ndim < score.ndim:
                sigma = sigma.unsqueeze(-1)

        w_x, w_y = self.sde.weights(t)
        if not torch.is_tensor(w_x):
            w_x = torch.full_like(t, float(w_x))
        if not torch.is_tensor(w_y):
            w_y = torch.full_like(t, float(w_y))
        w_x = w_x.to(device=score.device, dtype=score.real.dtype)
        w_y = w_y.to(device=score.device, dtype=score.real.dtype)
        while w_x.ndim < score.ndim:
            w_x = w_x.unsqueeze(-1)
        while w_y.ndim < score.ndim:
            w_y = w_y.unsqueeze(-1)

        mean_hat = x_t + sigma.square() * score
        x0_hat = (mean_hat - w_y * y_main) / w_x.clamp_min(1e-5)
        return score, x_t, y_main, t, sigma, w_x, w_y, x0_hat

    def _rrsd_score_to_x0(
        self,
        x_t: torch.Tensor,
        y_main: torch.Tensor,
        t: torch.Tensor,
        score: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Invert the OUVE marginal mean for the full-speech RRSD path.

        Uses the SDE's actual weights so the routing geometry remains consistent
        with the configured OUVE path. Returns (x0_base, sigma, w_x, w_y).
        """
        if self.sde.__class__.__name__ != "OUVESDE":
            raise ValueError("RRSD V1 currently supports OUVESDE only.")
        if t.ndim == 0:
            t_vec = torch.full((x_t.shape[0],), float(t.item()), device=x_t.device, dtype=torch.float32)
        else:
            t_vec = t.to(device=x_t.device, dtype=torch.float32).flatten()
        real_dtype = score.real.dtype if torch.is_complex(score) else score.dtype
        sigma = self.sde._std(t_vec).to(device=x_t.device, dtype=real_dtype)
        w_x, w_y = self.sde.weights(t_vec)
        if not torch.is_tensor(w_x):
            w_x = torch.full_like(t_vec, float(w_x))
        if not torch.is_tensor(w_y):
            w_y = torch.full_like(t_vec, float(w_y))
        w_x = w_x.to(device=x_t.device, dtype=real_dtype)
        w_y = w_y.to(device=x_t.device, dtype=real_dtype)
        while sigma.ndim < score.ndim:
            sigma = sigma.unsqueeze(-1)
        while w_x.ndim < score.ndim:
            w_x = w_x.unsqueeze(-1)
        while w_y.ndim < score.ndim:
            w_y = w_y.unsqueeze(-1)
        mean_hat = x_t + sigma.square() * score
        x0_base = (mean_hat - w_y * y_main) / w_x.clamp_min(1e-5)
        return x0_base, sigma, w_x, w_y

    def _rrsd_reverse_mean_step(
        self,
        *,
        x: torch.Tensor,
        y_main: torch.Tensor,
        t: torch.Tensor,
        score: torch.Tensor,
        dt: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Deterministic reverse-SDE mean step used only by the K-step teacher.

        This matches the mean of ``ReverseDiffusionPredictor`` for OUVESDE:

            x_next_mean = x - f_forward * dt + g(t)^2 * dt * score.

        No stochastic predictor noise and no corrector are used.  The purpose is
        to obtain a stable counterfactual teacher target, not to reproduce the
        entire stochastic sampler during training.
        """
        if self.sde.__class__.__name__ != "OUVESDE":
            raise ValueError("RRSD K-step teacher currently supports OUVESDE only.")
        t = t.to(device=x.device, dtype=torch.float32).flatten()
        dt = dt.to(device=x.device, dtype=torch.float32).flatten().clamp_min(0.0)
        drift, diffusion = self.sde.sde(x, y_main, t)
        dt_map = dt.view(-1, *([1] * (x.ndim - 1))).to(dtype=x.real.dtype if torch.is_complex(x) else x.dtype)
        g2_map = diffusion.to(device=x.device, dtype=dt_map.dtype).square().view(
            -1, *([1] * (x.ndim - 1))
        )
        x_next = x - drift * dt_map + g2_map * dt_map * score
        t_next = (t - dt).clamp_min(float(self.t_eps))
        return x_next, t_next

    def _rrsd_lookahead_teacher(
        self,
        *,
        target: torch.Tensor,
        x0_front: torch.Tensor,
        x0_base: torch.Tensor,
        x_t: torch.Tensor,
        y_main: torch.Tensor,
        y_cond: torch.Tensor,
        t: torch.Tensor,
        score_base: torch.Tensor,
        sigma: torch.Tensor,
        w_x: torch.Tensor,
        alpha_projection: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Vectorized trajectory-aware discrete K-step teacher.

        This is mathematically the same discrete teacher as the earlier K-step
        implementation.  The optimization is computational: candidate branches
        are stacked along the batch dimension, so K=1 with M candidates uses one
        enlarged frozen-model re-score instead of M serial re-scores.  For K>1,
        each lookahead depth uses one enlarged model forward.

        Candidate c applies

            alpha_c(f,tau) = c * alpha_projection(f,tau)

        only on the first transition.  The remaining K-1 transitions are
        base-only.  At the final state the frozen score network is evaluated and
        the candidate with smallest future x0 MSE is selected per utterance.
        """
        if not bool(getattr(self, "rrsd_freeze_base", False)):
            raise RuntimeError("RRSD K-step teacher requires a frozen base model.")

        B = int(target.shape[0])
        reduce_dims = tuple(range(1, target.ndim))
        candidates = tuple(float(v) for v in self.rrsd_lookahead_candidates)
        M = len(candidates)
        base_dt = float(self.rrsd_lookahead_dt)
        if base_dt <= 0.0:
            denom = max(int(getattr(self.sde, "N", 30)) - 1, 1)
            base_dt = (float(getattr(self.sde, "T", 1.0)) - float(self.t_eps)) / float(denom)
        base_dt = max(base_dt, 1e-6)

        full_size = (int(x0_base.shape[-2]), int(x0_base.shape[-1]))
        projection_full = F.interpolate(
            alpha_projection.float(), size=full_size, mode="bilinear", align_corners=False
        ).to(dtype=x0_base.real.dtype if torch.is_complex(x0_base) else x0_base.dtype)
        front_direction = x0_front - x0_base

        # 0 means all candidates in one enlarged batch. A positive chunk is a
        # memory fallback, not a different teacher.
        chunk = int(self.rrsd_lookahead_chunk)
        if chunk <= 0:
            chunk = M

        error_chunks = []
        with torch.no_grad():
            for c0 in range(0, M, chunk):
                cvals = candidates[c0:c0 + chunk]
                C = len(cvals)
                c_tensor = torch.tensor(cvals, device=x_t.device, dtype=torch.float32).view(C, 1, 1, 1, 1)

                alpha_maps = (c_tensor * projection_full.float().unsqueeze(0)).clamp(0.0, 1.0)
                alpha_maps = alpha_maps.to(dtype=front_direction.real.dtype if torch.is_complex(front_direction) else front_direction.dtype)
                delta_x0 = alpha_maps * front_direction.unsqueeze(0)  # [C,B,Cx,F,T]

                # Candidate-major layout [candidate, batch, ...] -> [C*B, ...].
                score_rep = score_base.unsqueeze(0).expand(C, *score_base.shape).reshape(C * B, *score_base.shape[1:])
                sigma_rep = sigma.unsqueeze(0).expand(C, *sigma.shape).reshape(C * B, *sigma.shape[1:])
                wx_rep = w_x.unsqueeze(0).expand(C, *w_x.shape).reshape(C * B, *w_x.shape[1:])
                score_first = score_rep + wx_rep * delta_x0.reshape(C * B, *delta_x0.shape[2:]) / sigma_rep.square().clamp_min(1e-8)

                x_cur = x_t.unsqueeze(0).expand(C, *x_t.shape).reshape(C * B, *x_t.shape[1:])
                y_main_cur = y_main.unsqueeze(0).expand(C, *y_main.shape).reshape(C * B, *y_main.shape[1:])
                y_cond_cur = y_cond.unsqueeze(0).expand(C, *y_cond.shape).reshape(C * B, *y_cond.shape[1:])
                t_cur = t.to(device=x_t.device, dtype=torch.float32).flatten().unsqueeze(0).expand(C, B).reshape(C * B)

                dt_cur = torch.minimum(
                    torch.full_like(t_cur, base_dt),
                    (t_cur - float(self.t_eps)).clamp_min(0.0),
                )
                x_cur, t_cur = self._rrsd_reverse_mean_step(
                    x=x_cur, y_main=y_main_cur, t=t_cur, score=score_first, dt=dt_cur
                )

                for _ in range(1, int(self.rrsd_lookahead_k)):
                    score_cur = self(x_cur, y_main_cur, t_cur, y_cond=y_cond_cur)
                    dt_cur = torch.minimum(
                        torch.full_like(t_cur, base_dt),
                        (t_cur - float(self.t_eps)).clamp_min(0.0),
                    )
                    x_cur, t_cur = self._rrsd_reverse_mean_step(
                        x=x_cur, y_main=y_main_cur, t=t_cur, score=score_cur, dt=dt_cur
                    )

                score_final = self(x_cur, y_main_cur, t_cur, y_cond=y_cond_cur)
                x0_final, _, _, _ = self._rrsd_score_to_x0(
                    x_cur, y_main_cur, t_cur, score_final
                )
                target_rep = target.unsqueeze(0).expand(C, *target.shape).reshape(C * B, *target.shape[1:])
                err = torch.abs(x0_final - target_rep).float().square().mean(dim=reduce_dims)
                error_chunks.append(err.view(C, B).transpose(0, 1))  # [B,C]

        error_matrix = torch.cat(error_chunks, dim=1)  # [B,M]
        best_index = error_matrix.argmin(dim=1)
        candidate_tensor = torch.tensor(candidates, device=target.device, dtype=torch.float32)
        best_scale = candidate_tensor[best_index]

        base_index = candidates.index(0.0)
        base_error = error_matrix[:, base_index]
        best_error = error_matrix.gather(1, best_index[:, None]).squeeze(1)
        advantage = (base_error - best_error).clamp_min(0.0)
        advantage_ratio = advantage / base_error.clamp_min(float(self.rrsd_oracle_eps))

        alpha_target = alpha_projection * best_scale.view(B, 1, 1, 1)
        stats = {
            "best_scale": best_scale.detach(),
            "best_index": best_index.detach(),
            "base_future_error": base_error.detach(),
            "best_future_error": best_error.detach(),
            "advantage": advantage.detach(),
            "advantage_ratio": advantage_ratio.detach(),
            "error_matrix": error_matrix.detach(),
        }
        return alpha_target.detach(), stats

    def _rrsd_future_grad_loss(
        self,
        *,
        target: torch.Tensor,
        x0_front: torch.Tensor,
        step_out: dict,
        pred: Dict[str, torch.Tensor],
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Direct one-step future re-score objective for the router.

        Q_phi emits one routing action (or g*a in gate mode). That action changes
        the reverse state once. The changed state is then passed through the
        *frozen* diffusion network one additional time. Model parameters remain
        frozen, but autograd through the network input is intentionally kept so
        the future loss can train Q_phi through ds_theta(x_next)/dx_next.

        There is no discrete oracle search in this mode.
        """
        if not bool(getattr(self, "rrsd_freeze_base", False)):
            raise RuntimeError("RRSD future_grad requires a frozen base model.")

        score, x_t, y_main, t, sigma, w_x, _, x0_base = self._score_step_geometry(step_out)
        y_cond = step_out.get("y_cond", None)
        if y_cond is None:
            y_cond = y_main

        # Freeze the evaluator parameters/state, not the computational path from
        # x_next back to the router. These tensors are constants for Q_phi.
        score_base = score.detach()
        x_t_det = x_t.detach()
        y_main_det = y_main.detach()
        y_cond_det = y_cond.detach()
        t_det = t.detach().to(device=x_t.device, dtype=torch.float32).flatten()
        sigma_det = sigma.detach()
        w_x_det = w_x.detach()
        target_det = target.detach()
        x0_base_det = x0_base.detach()

        base_dt = float(self.rrsd_lookahead_dt)
        if base_dt <= 0.0:
            denom = max(int(getattr(self.sde, "N", 30)) - 1, 1)
            base_dt = (float(getattr(self.sde, "T", 1.0)) - float(self.t_eps)) / float(denom)
        base_dt = max(base_dt, 1e-6)
        dt_cur = torch.minimum(
            torch.full_like(t_det, base_dt),
            (t_det - float(self.t_eps)).clamp_min(0.0),
        )

        # pred['delta_x0'] carries gradient to the router. Everything else is a
        # frozen geometric conversion from routed x0 direction to routed score.
        score_route = score_base + w_x_det * pred["delta_x0"] / sigma_det.square().clamp_min(1e-8)

        # Compare routed and no-route actions at the SAME future time t-dt.
        # The base branch is the counterfactual alpha=0 action.  This removes
        # the old cross-time normalization E_route(t-dt) / E_base(t), whose
        # denominator could reweight samples according to an unrelated current-
        # state reconstruction error.
        x_next_route, t_next = self._rrsd_reverse_mean_step(
            x=x_t_det, y_main=y_main_det, t=t_det, score=score_route, dt=dt_cur
        )
        x_next_base, t_next_base = self._rrsd_reverse_mean_step(
            x=x_t_det, y_main=y_main_det, t=t_det, score=score_base, dt=dt_cur
        )

        # Both branches share the same frozen evaluator. Stack them along the
        # batch dimension so the revised objective still needs only ONE extra
        # score-network forward per training step (at 2B batch size).
        x_next_both = torch.cat([x_next_route, x_next_base], dim=0)
        y_main_both = torch.cat([y_main_det, y_main_det], dim=0)
        y_cond_both = torch.cat([y_cond_det, y_cond_det], dim=0)
        t_next_both = torch.cat([t_next, t_next_base], dim=0)

        # IMPORTANT: no torch.no_grad() here. dnn parameters have requires_grad=False
        # under --rrsd_freeze_base, yet autograd must retain
        # d score_next_route / d x_next_route so L_future can train Q_phi.
        score_next_both = self(
            x_next_both, y_main_both, t_next_both, y_cond=y_cond_both
        )
        B = int(target_det.shape[0])
        score_next_route = score_next_both[:B]
        score_next_base = score_next_both[B:]

        x0_next_route, _, _, _ = self._rrsd_score_to_x0(
            x_next_route, y_main_det, t_next, score_next_route
        )
        x0_next_base, _, _, _ = self._rrsd_score_to_x0(
            x_next_base, y_main_det, t_next_base, score_next_base
        )

        reduce_dims = tuple(range(1, target_det.ndim))
        future_err = torch.abs(x0_next_route - target_det).float().square().mean(dim=reduce_dims)
        base_future_err = torch.abs(x0_next_base - target_det).float().square().mean(dim=reduce_dims)

        # Optional speaker-aware PLM contrastive regularizer on the ROUTED future
        # x0 estimate. This path remains differentiable:
        # alpha -> x_next_route -> frozen score net -> x0_next_route -> iSTFT ->
        # frozen UniSpeech-SAT -> InfoNCE -> router.
        loss_route_contrastive = torch.zeros(
            (), device=target.device, dtype=torch.float32
        )
        if bool(getattr(self, "rrsd_route_contrastive", False)):
            loss_route_contrastive = self._compute_contrastive_loss(
                x0_next_route,
                target_det,
                step_out.get("spk_ids", None),
                weight_clean=self.contrastive_clean_weight,
                weight_speaker=self.contrastive_speaker_weight,
            ).float()

        # Same-next-state relative objective:
        #   < 1 : routing improves the next-state x0 estimate
        #   = 1 : no benefit over alpha=0
        #   > 1 : routing hurts the next-state x0 estimate
        # Detach the denominator explicitly: it is a reference/control branch,
        # not an optimization target for the router.
        loss_future = (
            future_err / base_future_err.detach().clamp_min(float(self.rrsd_oracle_eps))
        ).mean()

        alpha_tile = pred["alpha_tile"].float()
        loss_smooth = torch.zeros((), device=target.device, dtype=torch.float32)
        if alpha_tile.shape[-2] > 1:
            loss_smooth = loss_smooth + alpha_tile.diff(dim=-2).abs().mean()
        if alpha_tile.shape[-1] > 1:
            loss_smooth = loss_smooth + alpha_tile.diff(dim=-1).abs().mean()

        loss_gate_cost = torch.zeros((), device=target.device, dtype=torch.float32)
        if self.rrsd_use_gate:
            loss_gate_cost = pred["gate_tile"].float().mean()

        loss = (
            float(self.rrsd_future_weight) * loss_future
            + float(self.rrsd_smooth_weight) * loss_smooth
            + float(self.rrsd_future_gate_cost) * loss_gate_cost
            + float(self.contrastive_weight) * loss_route_contrastive
        )

        with torch.no_grad():
            stats = {
                "rrsd_loss_future": loss_future.detach(),
                "rrsd_future_x0_error": future_err.mean(),
                "rrsd_base_future_x0_error": base_future_err.mean(),
                "rrsd_future_error_over_base_future": loss_future.detach(),
                "rrsd_route_contrastive_loss": loss_route_contrastive.detach(),
                "rrsd_route_contrastive_weighted": (
                    float(self.contrastive_weight) * loss_route_contrastive.detach()
                ),
                "rrsd_future_improve_fraction": (future_err < base_future_err).float().mean(),
                "rrsd_future_relative_error_change": (
                    (future_err - base_future_err)
                    / base_future_err.clamp_min(float(self.rrsd_oracle_eps))
                ).mean(),
                "rrsd_loss_smooth": loss_smooth.detach(),
                "rrsd_future_gate_cost": loss_gate_cost.detach(),
                "rrsd_alpha_mean": pred["alpha_tile"].detach().float().mean(),
                "rrsd_alpha_active_fraction": (pred["alpha_tile"].detach().float() > 1e-3).float().mean(),
                "rrsd_t_mean": t_det.float().mean(),
                "rrsd_next_t_mean": t_next.detach().float().mean(),
            }
            if self.rrsd_use_gate:
                stats.update({
                    "rrsd_gate_mean": pred["gate_tile"].detach().float().mean(),
                    "rrsd_gate_active_fraction": (pred["gate_tile"].detach().float() > 0.5).float().mean(),
                    "rrsd_strength_mean": pred["strength_tile"].detach().float().mean(),
                })
        return loss, stats

    def _compute_rrsd_loss(
        self,
        *,
        target: torch.Tensor,
        x0_front: torch.Tensor,
        step_out: dict,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Train RRSD with either the projection or trajectory-aware teacher.

        Modes
        -----
        projection + no gate:
            Backward-compatible RRSD V1.

        kstep + no gate:
            Keep one effective alpha output, but replace the target with
            ``alpha_K = c*_K * alpha_projection`` where ``c*_K`` is selected by
            K-step deterministic counterfactual rollout and re-scoring.

        projection + gate:
            Use the same shared backbone with two heads. ``g`` predicts whether
            local projection benefit is large enough to justify intervention;
            ``a`` predicts the conditional projection strength.

        kstep + gate:
            The same two-head router, but ``g`` additionally requires positive
            K-step trajectory advantage and ``a`` learns the K-step-rescaled
            projection strength. Inference always uses ``alpha_eff = g * a``.
        """
        zero = torch.zeros((), device=target.device, dtype=torch.float32)
        if self.rrsd_router is None or self.rrsd_weight <= 0.0:
            return zero, {}

        score, x_t, y_main, t, sigma, w_x, w_y, x0_base = self._score_step_geometry(step_out)
        y_cond = step_out.get("y_cond", None)
        if y_cond is None:
            y_cond = y_main

        score_det = score.detach()
        x0_base = x0_base.detach()
        x0_front = x0_front.detach().to(dtype=x0_base.dtype)
        target_det = target.detach()
        x_t_det = x_t.detach()
        y_main_det = y_main.detach()
        y_cond_det = y_cond.detach()
        t_det = t.detach()
        sigma_det = sigma.detach()
        w_x_det = w_x.detach()

        # The deployable router never receives clean speech.
        pred = self.rrsd_router(
            x_t=x_t_det,
            x0_base=x0_base,
            x0_front=x0_front,
            y_main=y_main_det,
            y_cond=y_cond_det,
            t=t_det,
            t_max=float(getattr(self.sde, "T", 1.0)),
            detach_inputs=True,
            scale=1.0,
        )

        # Direct future-gradient mode deliberately bypasses all clean-referenced
        # projection/K-step distillation targets. The future re-score loss is the
        # training signal for the router itself.
        if self.rrsd_teacher == "future_grad":
            return self._rrsd_future_grad_loss(
                target=target, x0_front=x0_front, step_out=step_out, pred=pred
            )

        # Local projection geometry is retained as the inexpensive band-time
        # proposal. K-step supervision decides how much of this proposal should
        # survive after looking at its future trajectory consequence.
        alpha_projection, projection_stats = solve_frontend_projection_oracle(
            target=target_det,
            x0_base=x0_base,
            x0_front=x0_front,
            num_bands=int(self.rrsd_router.num_bands),
            num_time_blocks=int(self.rrsd_router.num_time_blocks),
            eps=float(self.rrsd_oracle_eps),
        )

        trajectory_stats = None
        if self.rrsd_teacher == "kstep":
            alpha_teacher, trajectory_stats = self._rrsd_lookahead_teacher(
                target=target_det,
                x0_front=x0_front,
                x0_base=x0_base,
                x_t=x_t_det,
                y_main=y_main_det,
                y_cond=y_cond_det,
                t=t_det,
                score_base=score_det,
                sigma=sigma_det,
                w_x=w_x_det,
                alpha_projection=alpha_projection,
            )
        else:
            alpha_teacher = alpha_projection

        benefit_ratio = projection_stats["benefit_ratio"].clamp(0.0, 1.0)
        oracle_weight = 1.0 + float(self.rrsd_benefit_weight) * benefit_ratio

        loss_gate = zero
        loss_strength = zero
        gate_target = None

        if self.rrsd_use_gate:
            # g answers WHETHER routing is worth doing, while a answers HOW MUCH.
            #
            # Projection-gate mode (cheap): local projection benefit alone defines
            # the positive band-time tiles.  This directly tests whether explicitly
            # separating "whether" from "how much" fixes the almost-always-on
            # behaviour seen with the original single-alpha RRSD.
            #
            # K-step-gate mode (expensive): a tile must additionally belong to a
            # sample/timestep whose short counterfactual rollout has sufficient
            # future trajectory advantage.
            local_positive = (
                benefit_ratio > float(self.rrsd_gate_local_threshold)
            ).float() * (alpha_projection > 1e-3).float()
            if trajectory_stats is None:
                gate_target = local_positive.detach()
            else:
                future_positive = (
                    trajectory_stats["advantage_ratio"] > float(self.rrsd_gate_threshold)
                ).float().view(-1, 1, 1, 1)
                gate_target = (future_positive * local_positive).detach()

            # The strength teacher is the selected alpha teacher.  Under the
            # projection teacher this is alpha_projection; under K-step it is the
            # trajectory-rescaled alpha_K.  Strength is supervised only on tiles
            # that the gate teacher declares worth routing.
            strength_target = alpha_teacher.detach()
            effective_target = (gate_target * strength_target).detach()

            gate_pred = pred["gate_tile"].float()
            strength_pred = pred["strength_tile"].float()

            gate_bce = F.binary_cross_entropy(
                gate_pred, gate_target.float(), reduction="none"
            )
            gate_weight_map = 1.0 + (
                float(self.rrsd_gate_pos_weight) - 1.0
            ) * gate_target.float()
            loss_gate = (gate_bce * gate_weight_map).sum() / gate_weight_map.sum().clamp_min(1.0)

            strength_map = F.smooth_l1_loss(
                strength_pred, strength_target.float(), reduction="none"
            )
            strength_weight_map = gate_target.float() * oracle_weight
            loss_strength = (
                (strength_map * strength_weight_map).sum()
                / strength_weight_map.sum().clamp_min(1.0)
            )

            # Effective-alpha consistency also teaches negative tiles to become
            # true no-ops, which directly addresses cumulative over-routing.
            alpha_star = effective_target
        else:
            alpha_star = alpha_teacher.detach()

        loss_alpha_map = F.smooth_l1_loss(
            pred["alpha_tile"].float(), alpha_star.float(), reduction="none"
        )
        loss_alpha = (loss_alpha_map * oracle_weight).sum() / oracle_weight.sum().clamp_min(1.0)

        # Retained for projection/kstep distillation modes only. future_grad exits
        # above and therefore never mixes this immediate-x0 objective into its loss.
        reduce_dims = tuple(range(1, target_det.ndim))
        base_err = torch.abs(x0_base - target_det).float().square().mean(dim=reduce_dims)
        route_err = torch.abs(pred["x0_route"] - target_det).float().square().mean(dim=reduce_dims)
        loss_recon = (route_err / base_err.clamp_min(float(self.rrsd_oracle_eps))).mean()

        loss_smooth = torch.zeros((), device=target.device, dtype=torch.float32)
        a_eff = pred["alpha_tile"].float()
        if a_eff.shape[-2] > 1:
            loss_smooth = loss_smooth + a_eff.diff(dim=-2).abs().mean()
        if a_eff.shape[-1] > 1:
            loss_smooth = loss_smooth + a_eff.diff(dim=-1).abs().mean()

        loss = (
            loss_alpha
            + float(self.rrsd_recon_weight) * loss_recon
            + float(self.rrsd_smooth_weight) * loss_smooth
        )
        if self.rrsd_use_gate:
            loss = (
                loss
                + float(self.rrsd_gate_weight) * loss_gate
                + float(self.rrsd_strength_weight) * loss_strength
            )

        with torch.no_grad():
            oracle_active = (alpha_star > 1e-3).float().mean()
            oracle_front_heavy = (alpha_star > 0.5).float().mean()
            pred_active = (pred["alpha_tile"] > 1e-3).float().mean()
            stats = {
                "rrsd_loss_alpha": loss_alpha.detach(),
                "rrsd_loss_recon_norm": loss_recon.detach(),
                "rrsd_loss_smooth": loss_smooth.detach(),
                "rrsd_loss_gate": loss_gate.detach(),
                "rrsd_loss_strength": loss_strength.detach(),
                "rrsd_alpha_mean": pred["alpha_tile"].detach().float().mean(),
                "rrsd_alpha_target_mean": alpha_star.float().mean(),
                "rrsd_alpha_mae": torch.abs(
                    pred["alpha_tile"].detach().float() - alpha_star.float()
                ).mean(),
                "rrsd_alpha_active_fraction": pred_active,
                "rrsd_oracle_active_fraction": oracle_active,
                "rrsd_oracle_front_heavy_fraction": oracle_front_heavy,
                "rrsd_oracle_benefit_ratio_mean": benefit_ratio.mean(),
                "rrsd_route_improve_fraction": (route_err < base_err).float().mean(),
                "rrsd_route_relative_error_change": (
                    (route_err - base_err) / base_err.clamp_min(float(self.rrsd_oracle_eps))
                ).mean(),
                "rrsd_t_mean": t_det.float().mean(),
            }

            if trajectory_stats is not None:
                adv_ratio = trajectory_stats["advantage_ratio"].float()
                best_scale = trajectory_stats["best_scale"].float()
                stats.update({
                    "rrsd_kstep_best_scale_mean": best_scale.mean(),
                    "rrsd_kstep_scale_zero_fraction": (best_scale <= 1e-8).float().mean(),
                    "rrsd_kstep_advantage_ratio_mean": adv_ratio.mean(),
                    "rrsd_kstep_advantage_positive_fraction": (adv_ratio > 0.0).float().mean(),
                    "rrsd_kstep_gateworthy_fraction": (
                        adv_ratio > float(self.rrsd_gate_threshold)
                    ).float().mean(),
                    "rrsd_kstep_future_error_ratio": (
                        trajectory_stats["best_future_error"]
                        / trajectory_stats["base_future_error"].clamp_min(float(self.rrsd_oracle_eps))
                    ).mean(),
                })
                for candidate_scale in self.rrsd_lookahead_candidates:
                    tag = str(float(candidate_scale)).replace(".", "p").replace("-", "m")
                    stats[f"rrsd_kstep_scale_{tag}_fraction"] = (
                        (best_scale - float(candidate_scale)).abs() < 1e-6
                    ).float().mean()

            if self.rrsd_use_gate:
                stats.update({
                    "rrsd_gate_mean": pred["gate_tile"].detach().float().mean(),
                    "rrsd_gate_target_mean": gate_target.float().mean(),
                    "rrsd_gate_active_fraction": (
                        pred["gate_tile"].detach().float() > 0.5
                    ).float().mean(),
                    "rrsd_gate_target_active_fraction": (
                        gate_target.float() > 0.5
                    ).float().mean(),
                    "rrsd_projection_gateworthy_fraction": (
                        (benefit_ratio > float(self.rrsd_gate_local_threshold))
                        * (alpha_projection > 1e-3)
                    ).float().mean(),
                    "rrsd_strength_mean": pred["strength_tile"].detach().float().mean(),
                })
        return loss, stats

    def build_rrsd_score_calibrator(
        self,
        y_main: torch.Tensor,
        y_cond: Optional[torch.Tensor] = None,
        *,
        scale: float = 1.0,
    ):
        """Build the inference-time RRSD routing step used inside the PC sampler.

        Per predictor step:
          1) base score -> current x0_base
          2) Q_phi predicts alpha(t,f,tau)
          3) x0_route = x0_base + alpha * (x0_front-x0_base)
          4) x0_route is converted back to a routed score
          5) the ordinary reverse-SDE predictor integrates that routed score
        """
        if self.rrsd_router is None:
            return None
        if self.sde.__class__.__name__ != "OUVESDE":
            warnings.warn("RRSD V1 currently supports OUVESDE only; disabling routing.")
            return None
        if y_cond is None:
            y_cond = y_main

        # RRSD V1-clean routes the current diffusion x0 estimate toward the
        # *full-speech frontend estimate*.  This is exactly the candidate pair
        # validated by the frontend/backend Oracle Projection experiment.
        Y_pre, _ = self._main_and_cond_to_pre_noisy(y_main, y_cond)
        x0_front = Y_pre.detach()

        t_max = float(getattr(self.sde, "T", 1.0))
        scale = float(scale)

        def _score_calibrator(x_t, y_unused, t, score_base, step_index=None, num_steps=None):
            x0_base, sigma, w_x, _ = self._rrsd_score_to_x0(x_t, y_main, t, score_base)
            pred = self.rrsd_router(
                x_t=x_t,
                x0_base=x0_base,
                x0_front=x0_front,
                y_main=y_main,
                y_cond=y_cond,
                t=t,
                t_max=t_max,
                detach_inputs=True,
                scale=scale,
            )
            # mu(x0_route)-mu(x0_base) = w_x * delta_x0, hence the exact
            # corresponding score residual is w_x * delta_x0 / sigma^2.
            delta_score = w_x * pred["delta_x0"] / sigma.square().clamp_min(1e-8)
            score_route = score_base + delta_score.to(dtype=score_base.dtype)
            if torch.is_complex(score_route):
                score_route = torch.complex(
                    torch.nan_to_num(score_route.real),
                    torch.nan_to_num(score_route.imag),
                )
            else:
                score_route = torch.nan_to_num(score_route)

            # Lightweight inference diagnostics for optional debugging.
            self._last_rrsd_alpha_mean = pred["alpha_map"].detach().float().mean()
            self._last_rrsd_alpha_max = pred["alpha_map"].detach().float().max()
            return score_route

        return _score_calibrator

    def _compute_frontend_adaptive_score_guidance_loss(
        self,
        *,
        target: torch.Tensor,
        frontend_reference: torch.Tensor,
        step_out: dict,
    ):
        """Frontend-relative adaptive guidance directly in score space.

        This is the proposed Scheme-A extension of the existing regret-aware
        score-matching objective.  The original DSM loss is kept unchanged.
        A *detached* local reliability gate is computed from the relative clean
        reconstruction risk of the diffusion backend and the predictive frontend.
        Only where the backend is worse than the frontend by the configured
        margin do we add a score-space alignment term.

        Let F be the frontend reference and y the OUVE center.  The score that
        would make the current state ``x_t`` point exactly to F is

            s_F = (w_x F + w_y y - x_t) / sigma_t^2.

        The additional objective is

            L_guide = sum g * 0.5 * |sigma_t (s_theta - s_F)|^2,

        where g in [0,1] is a smooth function of positive *log regret*:

            r = [log(E_b/E_f) - log(1+margin)]_+,
            g = 1 - exp(-r / temperature).

        Thus the frontend is not a permanent teacher.  When the diffusion
        backend is already at least as good as the frontend, g=0 and training is
        exactly the original score-matching objective.  All gate quantities and
        the frontend target are detached, so gradients update the score backend
        only and cannot game the reliability detector.
        """
        if self.loss_type != "score_matching" or self.loss_weighting != "sigma^2":
            raise RuntimeError("adaptive_score_guidance requires score_matching with sigma^2 weighting.")
        if str(getattr(self, "frontend_nr_mode", "patch")).lower() != "patch":
            raise RuntimeError("adaptive_score_guidance requires --frontend_nr_mode patch.")
        if target.ndim != 4:
            raise RuntimeError(
                f"adaptive_score_guidance expects [B,C,F,T], got {tuple(target.shape)}"
            )

        eps = float(self.frontend_nr_eps)
        score, x_t, y_main, t, sigma, w_x, w_y, x0_backend = self._score_step_geometry(step_out)
        time_active = t <= float(self.frontend_nr_t_max)
        time_mask = time_active[:, None, None, None]

        # ------------------------------------------------------------------
        # 1) Detached frontend-relative reliability gate in x0 space.
        # ------------------------------------------------------------------
        with torch.no_grad():
            target_det = target.detach()
            front_det = frontend_reference.detach()
            x0_det = x0_backend.detach()

            backend_tf = 0.5 * torch.abs(x0_det - target_det).float().square().sum(dim=1, keepdim=True)
            front_tf = 0.5 * torch.abs(front_det - target_det).float().square().sum(dim=1, keepdim=True)

            kf = int(self.frontend_nr_patch_freq)
            kt = int(self.frontend_nr_patch_time)
            pool_kwargs = dict(
                kernel_size=(kf, kt),
                stride=1,
                padding=(kf // 2, kt // 2),
                count_include_pad=False,
            )
            backend_patch = F.avg_pool2d(backend_tf, **pool_kwargs)
            front_patch = F.avg_pool2d(front_tf, **pool_kwargs)

            # Robust denominator: preserve the existing near-silence protection.
            front_mean = front_patch.mean(dim=(2, 3), keepdim=True)
            ratio_floor = (float(self.frontend_nr_floor_ratio) * front_mean).clamp_min(eps)
            ratio_denom = torch.maximum(front_patch, ratio_floor)

            # Log ratio is better behaved than a raw ratio when E_f is tiny.
            log_ratio = torch.log(backend_patch + eps) - torch.log(ratio_denom + eps)
            log_margin = math.log1p(float(self.frontend_nr_margin))
            positive_log_regret = F.relu(log_ratio - log_margin)
            positive_log_regret = positive_log_regret * time_mask

            temperature = float(self.frontend_nr_guidance_temperature)
            gate = 1.0 - torch.exp(-positive_log_regret / temperature)

            # Optional tail focusing for an apples-to-apples comparison with the
            # original hard top-k regret mining.  Default=1.0 is fully soft.
            tail_fraction = float(self.frontend_nr_guidance_tail_fraction)
            regret_flat = positive_log_regret.flatten(1)
            n_local = regret_flat.shape[1]
            tail_n = max(1, min(n_local, int(math.ceil(tail_fraction * n_local))))
            if tail_n < n_local:
                top_idx = torch.topk(regret_flat, k=tail_n, dim=1, largest=True).indices
                keep = torch.zeros_like(regret_flat)
                keep.scatter_(1, top_idx, 1.0)
                gate = gate * keep.view_as(gate)
            else:
                keep = torch.ones_like(regret_flat)

            gate = gate.detach()
            local_ratio = torch.exp(log_ratio).clamp_max(1e6)
            local_violation = positive_log_regret > 0.0

        # ------------------------------------------------------------------
        # 2) Convert the frontend x0 reference into a *score target* at the
        #    exact same x_t and t.  This is the key difference from output KD.
        # ------------------------------------------------------------------
        front_ref = frontend_reference.detach().to(device=score.device, dtype=score.dtype)
        y_det = y_main.detach()
        x_t_det = x_t.detach()
        mean_front = w_x.to(score.dtype) * front_ref + w_y.to(score.dtype) * y_det
        score_front = (mean_front - x_t_det) / sigma.square().clamp_min(eps)
        score_front = score_front.detach()

        guidance_tf = 0.5 * torch.abs(
            sigma * (score - score_front)
        ).float().square().sum(dim=1, keepdim=True)
        gate_f = gate.to(guidance_tf.dtype)
        weighted_guidance_sum = (guidance_tf * gate_f).sum()
        coverage = gate_f.mean().detach()
        coverage_target = float(self.frontend_nr_guidance_coverage_target)
        if coverage_target > 0.0:
            # Calibrate the sparse gate toward a fixed effective coverage so the
            # global loss magnitude does not collapse merely because only a small
            # fraction of TF bins violate the frontend-relative no-regret condition.
            coverage_scale = (
                torch.as_tensor(coverage_target, device=gate_f.device, dtype=gate_f.dtype)
                / coverage.clamp_min(eps)
            ).clamp(max=10.0)
        else:
            coverage_scale = torch.ones((), device=gate_f.device, dtype=gate_f.dtype)
        guidance_loss = weighted_guidance_sum * coverage_scale

        with torch.no_grad():
            score_loss_tf = 0.5 * torch.abs(
                score * sigma + step_out["z"].to(score.dtype)
            ).float().square().sum(dim=1, keepdim=True)
            raw_score_loss = score_loss_tf.sum().clamp_min(eps)
            active_denom = time_active.float().sum().clamp_min(1.0)
            sample_has_violation = local_violation.flatten(1).any(dim=1).float()
            local_active = local_violation.float().mean(dim=(1, 2, 3))
            local_ratio_mean = local_ratio.mean(dim=(1, 2, 3))
            local_ratio_max = local_ratio.flatten(1).max(dim=1).values

            def _active_mean(v: torch.Tensor) -> torch.Tensor:
                return (v.float() * time_active.float()).sum() / active_denom

            stats = {
                "frontend_nr_time_active_fraction": time_active.float().mean(),
                "frontend_nr_active_fraction": _active_mean(sample_has_violation),
                "frontend_nr_local_active_fraction": _active_mean(local_active),
                "frontend_nr_diff_front_ratio": _active_mean(local_ratio_mean),
                "frontend_nr_local_max_ratio": _active_mean(local_ratio_max),
                "frontend_guidance_gate_mean": gate.mean(),
                "frontend_guidance_gate_max": gate.max(),
                "frontend_guidance_gate_active_fraction": (gate > 0.0).float().mean(),
                "frontend_guidance_coverage": coverage,
                "frontend_guidance_coverage_scale": coverage_scale.detach(),
                "frontend_guidance_loss_to_raw_score_ratio": guidance_loss.detach() / raw_score_loss,
                "frontend_guidance_score_distance_mean": guidance_tf.mean(),
            }

        return guidance_loss.float(), stats

    def _compute_frontend_regret_score_reweighted_loss(
        self,
        *,
        target: torch.Tensor,
        frontend_reference: torch.Tensor,
        step_out: dict,
    ):
        """Reweight the MAIN score-matching TF loss using local frontend regret.

        The detector is target-derived and detached:
            E_b(f,t) = local error of the current backend x0 estimate
            E_f(f,t) = local error of the frontend reference
            severity = relu(E_b / stabilized(E_f) - (1 + margin))

        Only t <= frontend_nr_t_max is eligible.  We optionally keep the top-k
        severe patches, normalize severity over positive patches per sample, and
        form a bounded multiplier

            w(f,t) = 1 + strength * normalized_severity,

        clipped by frontend_nr_reweight_max.  The original score-matching loss
        |score*sigma + z|^2 is then multiplied by w at the same TF locations.

        Crucially, ``w`` is detached.  Gradients therefore flow only through the
        ordinary score loss; the degradation detector does not create a second
        optimization path or an extra score forward.
        """
        if self.loss_type != "score_matching" or self.loss_weighting != "sigma^2":
            raise RuntimeError("score_reweight requires score_matching with sigma^2 weighting.")
        if str(getattr(self, "frontend_nr_mode", "patch")).lower() != "patch":
            raise RuntimeError("score_reweight currently requires --frontend_nr_mode patch.")
        if target.ndim != 4:
            raise RuntimeError(f"score_reweight expects [B,C,F,T], got {tuple(target.shape)}")

        eps = float(self.frontend_nr_eps)
        t = step_out["t"].detach().to(device=target.device, dtype=torch.float32).flatten()
        time_active = t <= float(self.frontend_nr_t_max)
        time_mask = time_active[:, None, None, None]

        # x_hat_spec is produced by the SAME main score forward used by base_loss.
        x0_backend = step_out.get("x_hat_spec", None)
        if x0_backend is None:
            raise RuntimeError("score_reweight requires return_details=True so x_hat_spec is available.")

        with torch.no_grad():
            x0_det = x0_backend.detach()
            target_det = target.detach()
            ref_det = frontend_reference.detach()

            backend_tf = 0.5 * torch.abs(x0_det - target_det).float().square().sum(dim=1, keepdim=True)
            front_tf = 0.5 * torch.abs(ref_det - target_det).float().square().sum(dim=1, keepdim=True)

            kf = int(self.frontend_nr_patch_freq)
            kt = int(self.frontend_nr_patch_time)
            pool_kwargs = dict(
                kernel_size=(kf, kt),
                stride=1,
                padding=(kf // 2, kt // 2),
                count_include_pad=False,
            )
            backend_patch = F.avg_pool2d(backend_tf, **pool_kwargs)
            front_patch = F.avg_pool2d(front_tf, **pool_kwargs)

            front_mean = front_patch.mean(dim=(2, 3), keepdim=True)
            ratio_floor = (float(self.frontend_nr_floor_ratio) * front_mean).clamp_min(eps)
            ratio_denom = torch.maximum(front_patch, ratio_floor)
            local_ratio = backend_patch / ratio_denom
            margin_factor = 1.0 + float(self.frontend_nr_margin)
            severity = F.relu(local_ratio - margin_factor)
            severity = severity * time_mask
            raw_positive = severity > 0.0

            # Keep only the most severe fraction. Zeros stay zeros, so when the
            # true violation rate is below top-k no false-positive patch is boosted.
            sev_flat = severity.flatten(1)
            n_local = sev_flat.shape[1]
            topk_n = max(1, min(n_local, int(math.ceil(float(self.frontend_nr_topk) * n_local))))
            if topk_n < n_local:
                topk_idx = torch.topk(sev_flat, k=topk_n, dim=1, largest=True).indices
                keep = torch.zeros_like(sev_flat)
                keep.scatter_(1, topk_idx, 1.0)
                sev_flat = sev_flat * keep
            else:
                keep = torch.ones_like(sev_flat)
            severity = sev_flat.view_as(severity)

            positive = severity > 0.0
            positive_count = positive.float().sum(dim=(1, 2, 3), keepdim=True)
            positive_mean = severity.sum(dim=(1, 2, 3), keepdim=True) / positive_count.clamp_min(1.0)
            # Samples with no violation keep severity_norm=0 and thus weight=1.
            severity_norm = torch.where(
                positive,
                severity / positive_mean.clamp_min(eps),
                torch.zeros_like(severity),
            )

            weight_map = 1.0 + float(self.frontend_nr_reweight_strength) * severity_norm
            weight_map = weight_map.clamp(max=float(self.frontend_nr_reweight_max))
            weight_map = weight_map.detach()

        score = step_out["forward_out"]
        z = step_out.get("z", None)
        if z is None:
            raise RuntimeError(
                "score_reweight requires step_out['z']; update sgmse/model.py so _step(return_details=True) returns z."
            )
        z = z.to(device=score.device, dtype=score.dtype)

        sigma_override = step_out.get("sigma_override", None)
        if sigma_override is None:
            sigma_xt = self.sde._std(t).to(device=score.device, dtype=score.real.dtype)
            sigma_xt = sigma_xt[:, None, None, None]
        else:
            sigma_xt = sigma_override.detach().to(device=score.device, dtype=score.real.dtype)
            while sigma_xt.ndim < score.ndim:
                sigma_xt = sigma_xt.unsqueeze(-1)

        # Same per-bin objective as ScoreModel._loss(..., score_matching, sigma^2).
        score_loss_tf = 0.5 * torch.abs(score * sigma_xt + z).float().square().sum(dim=1, keepdim=True)
        raw_reconstructed = score_loss_tf.sum()
        reweighted_loss = (score_loss_tf * weight_map.to(score_loss_tf.dtype)).sum()

        with torch.no_grad():
            active_denom = time_active.float().sum().clamp_min(1.0)
            local_violation = raw_positive.float()
            selected_violation = (severity > 0.0).float()
            sample_has_violation = local_violation.flatten(1).any(dim=1).float()
            local_active_per_sample = local_violation.mean(dim=(1, 2, 3))
            local_ratio_mean = local_ratio.mean(dim=(1, 2, 3))
            local_ratio_max = local_ratio.flatten(1).max(dim=1).values
            ratio_flat = local_ratio.flatten(1)
            keep_count = keep.sum(dim=1).clamp_min(1.0)
            local_topk_ratio = (ratio_flat * keep).sum(dim=1) / keep_count
            local_topk_active_fraction = (local_violation.flatten(1) * keep).sum(dim=1) / keep_count

            def _active_mean(v: torch.Tensor) -> torch.Tensor:
                if v.ndim == 0:
                    return v
                return (v.float() * time_active.float()).sum() / active_denom

            hard_score = (score_loss_tf.detach() * selected_violation).sum()
            total_score = score_loss_tf.detach().sum().clamp_min(eps)
            stats = {
                "frontend_nr_time_active_fraction": time_active.float().mean(),
                "frontend_nr_active_fraction": _active_mean(sample_has_violation),
                "frontend_nr_local_active_fraction": _active_mean(local_active_per_sample),
                "frontend_nr_diff_front_ratio": _active_mean(local_ratio_mean),
                "frontend_nr_local_max_ratio": _active_mean(local_ratio_max),
                "frontend_nr_local_topk_ratio": _active_mean(local_topk_ratio),
                "frontend_nr_local_topk_active_fraction": _active_mean(local_topk_active_fraction),
                "frontend_nr_reweight_mean": weight_map.mean(),
                "frontend_nr_reweight_max": weight_map.max(),
                "frontend_nr_reweight_boosted_fraction": (weight_map > 1.0).float().mean(),
                "frontend_nr_hard_score_loss_fraction": hard_score / total_score,
                "frontend_nr_reweighted_to_raw_ratio": reweighted_loss.detach() / raw_reconstructed.detach().clamp_min(eps),
                "frontend_nr_raw_reconstruction_ratio": raw_reconstructed.detach() / step_out["base_loss"].detach().abs().clamp_min(eps),
            }

        return reweighted_loss.float(), stats

    def _compute_frontend_no_regret_loss(
        self,
        target: torch.Tensor,
        frontend_reference: torch.Tensor,
        step_out: Dict[str, Any],
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Frontend-relative no-regret loss for the score backend.

        Two granularities are supported:

        ``global`` (legacy):
            Compare one reconstruction error per sample and penalize only when
            the backend is globally worse than the frontend by the margin.

        ``patch`` (recommended):
            1) Build backend/frontend squared-error maps in the STFT domain.
            2) Smooth them with a local F x T average-pooling window.
            3) Detect local degradation with the *relative* error ratio
                   E_backend_patch / max(E_front_patch, local_floor).
            4) Keep only the top-k most severe local regions.
            5) Apply the actual no-regret penalty in absolute error units:
                   ReLU(E_backend_patch - (1+margin) E_front_patch).

        The relative ratio is used only for robust region selection/diagnostics;
        the optimized penalty remains the original one-sided no-regret excess.
        This avoids exploding gradients in near-silent bins while making sparse
        local degradation visible even when global MSE is already better.

        DDP safety:
        ``t`` is sampled independently on every rank. Every rank therefore runs
        the same auxiliary score forward and returns the same metric keys. A rank
        with no t <= frontend_nr_t_max runs one dummy sample with zero weight.

        Inputs use the same coordinates as the current diffusion task:
          * normal task: target=X and frontend_reference=Y_pre;
          * residual task: target=X-Y_pre and frontend_reference=0.
        Detached states keep this auxiliary objective backend-only.
        """
        zero = torch.zeros((), device=target.device, dtype=torch.float32)
        if float(getattr(self, "frontend_nr_weight", 0.0)) <= 0.0:
            return zero, {}

        t_all = step_out["t"].detach().to(device=target.device, dtype=torch.float32).flatten()
        time_active_all = t_all <= float(self.frontend_nr_t_max)
        time_active_fraction = time_active_all.float().mean()
        active_count = time_active_all.float().sum()
        stats_denom = active_count.clamp_min(1.0)

        # IMPORTANT FOR DDP: every rank executes one NR score forward.
        active_idx = torch.nonzero(time_active_all, as_tuple=False).flatten()
        if active_idx.numel() == 0:
            active_idx = torch.zeros(1, device=t_all.device, dtype=torch.long)
            sample_weight = torch.zeros(1, device=target.device, dtype=torch.float32)
        else:
            sample_weight = torch.ones(active_idx.numel(), device=target.device, dtype=torch.float32)

        x_t = step_out["x_t"].index_select(0, active_idx).detach()
        t = t_all.index_select(0, active_idx)
        y_main = step_out["y"].index_select(0, active_idx).detach()
        y_cond_all = step_out.get("y_cond", None)
        y_cond = None if y_cond_all is None else y_cond_all.index_select(0, active_idx).detach()

        # Detached conditions => NR gradients update score backend only.
        score = self(x_t, y_main, t, y_cond=y_cond)

        # PathSim V-A may override perturbation sigma for the standard task.
        sigma_override_all = step_out.get("sigma_override", None)
        if sigma_override_all is None:
            x0_backend = self._ouve_score_to_x0(x_t, y_main, t, score)
        else:
            sigma_eff = sigma_override_all.index_select(0, active_idx).detach().to(
                device=x_t.device,
                dtype=x_t.real.dtype if torch.is_complex(x_t) else x_t.dtype,
            )
            while sigma_eff.ndim < x_t.ndim:
                sigma_eff = sigma_eff.unsqueeze(-1)
            alpha = self.sde.alpha(t).view(-1, 1, 1, 1).to(dtype=sigma_eff.dtype)
            mean_hat = x_t + sigma_eff.square() * score
            x0_backend = (mean_hat - (1.0 - alpha) * y_main) / alpha.clamp_min(1e-5)

        target_active = target.index_select(0, active_idx).detach()
        reference_active = frontend_reference.index_select(0, active_idx).detach()
        reduce_dims = tuple(range(1, target_active.ndim))
        eps = float(self.frontend_nr_eps)

        # Global errors are retained for diagnostics in both modes.
        backend_error = 0.5 * torch.abs(x0_backend - target_active).float().square().sum(dim=reduce_dims)
        front_error = 0.5 * torch.abs(reference_active - target_active).float().square().sum(dim=reduce_dims)
        target_energy = 0.5 * torch.abs(target_active).float().square().sum(dim=reduce_dims).clamp_min(eps)
        backend_rel = backend_error / target_energy
        front_rel = front_error / target_energy
        global_ratio = backend_error / front_error.clamp_min(eps)

        nr_mode = str(getattr(self, "frontend_nr_mode", "global")).lower()
        if nr_mode == "global":
            threshold = (1.0 + float(self.frontend_nr_margin)) * front_error
            violation = F.relu(backend_error - threshold)
            loss_nr = (violation * sample_weight).sum()

            stats = {
                "frontend_nr_time_active_fraction": time_active_fraction.detach(),
                "frontend_nr_active_fraction": (
                    ((violation > 0.0).float() * sample_weight).sum() / stats_denom
                ).detach(),
                "frontend_nr_diff_front_ratio": (
                    (global_ratio * sample_weight).sum() / stats_denom
                ).detach(),
                "frontend_nr_diff_rel_error": (
                    (backend_rel * sample_weight).sum() / stats_denom
                ).detach(),
                "frontend_nr_front_rel_error": (
                    (front_rel * sample_weight).sum() / stats_denom
                ).detach(),
                "frontend_nr_t_mean": ((t * sample_weight).sum() / stats_denom).detach(),
            }
            return loss_nr.float(), stats

        if nr_mode != "patch":
            raise RuntimeError(f"Unsupported frontend_nr_mode={nr_mode!r}")
        if target_active.ndim != 4:
            raise RuntimeError(
                "frontend_nr_mode=patch expects STFT tensors shaped [B,C,F,T], "
                f"got target shape={tuple(target_active.shape)}"
            )

        # ------------------------------------------------------------------
        # Fine-grained local no-regret
        # ------------------------------------------------------------------
        # Sum over complex/STFT channels but keep frequency and time.
        # Shape: [B, 1, F, T].
        backend_tf = 0.5 * torch.abs(x0_backend - target_active).float().square().sum(dim=1, keepdim=True)
        front_tf = 0.5 * torch.abs(reference_active - target_active).float().square().sum(dim=1, keepdim=True)

        kf = int(self.frontend_nr_patch_freq)
        kt = int(self.frontend_nr_patch_time)
        pool_kwargs = dict(
            kernel_size=(kf, kt),
            stride=1,
            padding=(kf // 2, kt // 2),
            count_include_pad=False,
        )
        backend_patch = F.avg_pool2d(backend_tf, **pool_kwargs)
        front_patch = F.avg_pool2d(front_tf, **pool_kwargs)

        # Relative detector used only to decide WHICH patches are severe.
        # A per-sample floor prevents silent/easy bins from creating huge ratios.
        front_mean = front_patch.mean(dim=(2, 3), keepdim=True)
        ratio_floor = (
            float(self.frontend_nr_floor_ratio) * front_mean
        ).clamp_min(eps)
        ratio_denom = torch.maximum(front_patch, ratio_floor)
        local_ratio = backend_patch / ratio_denom

        margin_factor = 1.0 + float(self.frontend_nr_margin)
        relative_severity = F.relu(local_ratio - margin_factor)

        # The optimized penalty keeps the original no-regret units/meaning.
        # The ratio above only supplies a robust local detector / hard miner.
        absolute_excess = F.relu(backend_patch - margin_factor * front_patch)

        severity_flat = relative_severity.flatten(1)
        excess_flat = absolute_excess.flatten(1)
        ratio_flat = local_ratio.flatten(1)
        n_local = severity_flat.shape[1]
        topk_n = max(1, min(n_local, int(math.ceil(float(self.frontend_nr_topk) * n_local))))

        # Hard-region mining is intentionally non-differentiable in selection;
        # gradients flow through the selected absolute excess values only.
        topk_idx = torch.topk(severity_flat.detach(), k=topk_n, dim=1, largest=True).indices
        selected_excess = torch.gather(excess_flat, 1, topk_idx)
        selected_severity = torch.gather(severity_flat, 1, topk_idx)
        selected_ratio = torch.gather(ratio_flat, 1, topk_idx)

        # Sum selected local excesses per sample. This preserves a scale closer
        # to the old summed global MSE than a dimensionless ratio-only loss.
        per_sample_loss = selected_excess.sum(dim=1)
        loss_nr = (per_sample_loss * sample_weight).sum()

        local_violation_mask = relative_severity > 0.0
        sample_has_violation = local_violation_mask.flatten(1).any(dim=1).float()
        local_active_fraction = local_violation_mask.float().mean(dim=(1, 2, 3))
        local_ratio_mean = local_ratio.mean(dim=(1, 2, 3))
        local_ratio_max = local_ratio.flatten(1).max(dim=1).values
        local_topk_ratio = selected_ratio.mean(dim=1)
        local_topk_active_fraction = (selected_severity > 0.0).float().mean(dim=1)

        def _active_mean(v: torch.Tensor) -> torch.Tensor:
            return (v.float() * sample_weight).sum() / stats_denom

        stats = {
            # Existing keys retained so old W&B dashboards still work.
            # In patch mode, active_fraction means "sample has >=1 local violation".
            "frontend_nr_time_active_fraction": time_active_fraction.detach(),
            "frontend_nr_active_fraction": _active_mean(sample_has_violation).detach(),
            "frontend_nr_diff_front_ratio": _active_mean(local_ratio_mean).detach(),
            "frontend_nr_diff_rel_error": _active_mean(backend_rel).detach(),
            "frontend_nr_front_rel_error": _active_mean(front_rel).detach(),
            "frontend_nr_t_mean": _active_mean(t).detach(),
            # New local diagnostics.
            "frontend_nr_local_active_fraction": _active_mean(local_active_fraction).detach(),
            "frontend_nr_local_max_ratio": _active_mean(local_ratio_max).detach(),
            "frontend_nr_local_topk_ratio": _active_mean(local_topk_ratio).detach(),
            "frontend_nr_local_topk_active_fraction": _active_mean(local_topk_active_fraction).detach(),
        }
        return loss_nr.float(), stats

    def training_step(self, batch, batch_idx):
        # batch: (x_wav, y_wav, spk_id)
        if len(batch) == 2:
            x_wav, y_wav = batch
            spk_ids = None
        else:
            x_wav, y_wav, spk_ids = batch
        '''
        # 保存一组样本，判断样本是否匹配
        if self.global_step == 0 and batch_idx == 0:
            save_dir = "./"
            os.makedirs(save_dir, exist_ok=True)

            x_wav, y_wav, spk_id = batch   # x=clean, y=noisy

            sr = getattr(self, "sr", 16000)

            # 只存 batch 里的第一个样本
            clean = x_wav[0].detach().cpu()
            noisy = y_wav[0].detach().cpu()

            if clean.dim() == 1:
                clean = clean.unsqueeze(0)
            if noisy.dim() == 1:
                noisy = noisy.unsqueeze(0)

            save(os.path.join(save_dir, "train_step0_batch0_clean.wav"), clean, sr)
            save(os.path.join(save_dir, "train_step0_batch0_noisy.wav"), noisy, sr)

            print(f"[debug] saved first training batch sample to {save_dir}")
        '''
        
        # ======== [DEBUG] 打印输入统计 ========
        # self._print_tensor_stats("x_wav(clean)", x_wav, batch_idx)
        # self._print_tensor_stats("y_wav(noisy)", y_wav, batch_idx)

        # ======== [DEBUG] 打印 SEMamba 输出 ========
        # self._print_tensor_stats("y_pre(SEMamba)", y_pre, batch_idx)


        # 频谱域：SGMSE 使用自己的 STFT + spec_fwd 得到时频谱
        X = self._sgmse_stft(x_wav)
        Y_noisy = self._sgmse_stft(y_wav)
        
        if self.frontend_name == "storm" and hasattr(self.frontend, "forward_spec"):
            # Reuse Y_noisy to avoid second STFT + spectral compression.
            # Frontend and data-module spectral configs must match.
            Y_pre = self.frontend.forward_transformed_spec(Y_noisy)
            # For spectral frontend supervision, waveform reconstruction is
            # unused. Avoid an unnecessary differentiable inverse STFT.
            needs_time_aux = (
                self.frontend_trainable
                and self.frontend_aux_loss in ("time_l1", "time_mse")
            )
            y_pre = None

        else:
            y_pre = self._frontend_forward(y_wav)
            y_pre = _match_length(y_pre, y_wav)
            Y_pre = self._sgmse_stft(y_pre)

        # =============== CTM distillation branch ===============
        if self.ctm_distill:
            # Determine main y / conditioning and discriminative prior xD in spec-domain
            if self.train_type == "mode3":
                y_main = Y_noisy
                y_cond = Y_pre
                xD = Y_pre
            else:  # mode2
                y_main = Y_pre
                y_cond = Y_noisy
                xD = Y_pre

            # Sample times: t in (t_eps, T), s in (tmin, t), u in [s,t)
            B = X.shape[0]
            device = X.device
            t = torch.rand(B, device=device) * (self.sde.T - self.t_eps) + self.t_eps
            t0 = torch.full_like(t, float(self.ctm_tmin))
            s = t0 + torch.rand(B, device=device) * (t - t0)
            if self.ctm_u_mode == "fixed":
                u = s + float(self.ctm_u_frac) * (t - s)
            else:
                u = s + torch.rand(B, device=device) * (t - s)

            # Sample x_t from SBVE marginal (uses clean X and y_main)
            mean, std = self.sde.marginal_prob(X, y_main, t)
            z = torch.randn_like(X)
            x_t = mean + std[:, None, None, None] * z

            # Teacher partial solve t->u (guided toward xD)
            x_u_teacher = self._teacher_partial_solve(x_t, y_main, t, u, y_cond=y_cond, xD=xD)

            # Target (EMA) generates endpoint target via two jumps: u->s, s->t0
            x_s_tgt = self._G_apply("target", x_u_teacher, y_main, u, s, y_cond=y_cond)
            x_0_tgt = self._G_apply("target", x_s_tgt, y_main, s, t0, y_cond=y_cond)

            # Student: one jump t->s, then target jump s->t0
            x_s_est = self._G_apply("student", x_t, y_main, t, s, y_cond=y_cond)
            x_0_est = self._G_apply("target", x_s_est, y_main, s, t0, y_cond=y_cond)

            # CTM loss (complex MSE)
            diff = x_0_est - x_0_tgt
            loss_ctm = torch.mean(0.5 * torch.sum(torch.abs(diff).reshape(B, -1) ** 2, dim=-1))

            # Optional: mix in base diffusion loss for stability
            loss_base = torch.tensor(0.0, device=device)
            if self.ctm_mix_base > 0.0:
                forward_out = self(x_t, y_main, t, y_cond=y_cond)
                loss_base = self._loss(forward_out, x_t, z, t, mean, X, y_main)

            # Optional prior loss: pull endpoint toward SEMamba estimate (stopgrad)
            loss_prior = torch.tensor(0.0, device=device)
            if self.ctm_prior_weight > 0.0:
                d2 = x_0_est - xD.detach()
                loss_prior = torch.mean(0.5 * torch.sum(torch.abs(d2).reshape(B, -1) ** 2, dim=-1))

            # Optional time-domain L1 (slow)
            loss_td = torch.tensor(0.0, device=device)
            if self.ctm_time_l1 > 0.0:
                x0_est_wav = self._spec_to_wave_batch(x_0_est)
                x0_tgt_wav = self._spec_to_wave_batch(x_0_tgt)
                loss_td = torch.mean(torch.abs(x0_est_wav - x0_tgt_wav))

            loss = self.ctm_weight * loss_ctm + self.ctm_mix_base * loss_base + self.ctm_prior_weight * loss_prior + self.ctm_time_l1 * loss_td

            self.log("train_loss", loss, on_step=True, on_epoch=True, sync_dist=True, prog_bar=True)
            self.log("ctm_loss", loss_ctm, on_step=True, on_epoch=True, sync_dist=True, prog_bar=False)
            if self.ctm_mix_base > 0.0:
                self.log("base_loss", loss_base, on_step=True, on_epoch=True, sync_dist=True, prog_bar=False)
            if self.ctm_prior_weight > 0.0:
                self.log("prior_loss", loss_prior, on_step=True, on_epoch=True, sync_dist=True, prog_bar=False)
            if self.ctm_time_l1 > 0.0:
                self.log("td_l1", loss_td, on_step=True, on_epoch=True, sync_dist=True, prog_bar=False)
            return loss

        # Build either the original full-speech diffusion task or the
        # predictive residual-anchored task. ScoreModel._step(mode2) swaps its
        # second/third tensors, so (R_target,R_obs,R_center) becomes
        # y_main=R_center and y_cond=R_obs as required by residual OUVE.
        loss_residual_init = torch.zeros((), device=X.device, dtype=torch.float32)
        residual_init_stats = {}
        r_target = None
        if self.residual_refine_enable:
            if self.use_contrastive and self.contrastive_weight > 0.0:
                raise ValueError("Disable contrastive loss for the first residual-refinement experiment.")
            r_target, r_obs, r_det, r_center = self._build_predictive_residual_state(
                X=X, Y_noisy=Y_noisy, Y_pre=Y_pre
            )
            if spk_ids is not None:
                step_out = self._step(
                    (r_target, r_obs, r_center, spk_ids), batch_idx, return_details=True
                )
            else:
                step_out = self._step(
                    (r_target, r_obs, r_center), batch_idx, return_details=True
                )
            loss_residual_init, residual_init_stats = self._compute_residual_initializer_loss(
                r_target=r_target, r_obs=r_obs, r_det=r_det
            )
            # Diagnostics for the learned residual anchor trajectory.  beta(t)
            # is the actual R_theta coefficient in the OUVE perturbation mean:
            #   m(t) = alpha(t) R_target + beta(t) R_theta.
            if hasattr(self.sde, "residual_anchor_lambda"):
                with torch.no_grad():
                    t_anchor = step_out["t"].detach()
                    lam = self.sde.residual_anchor_lambda(t_anchor).float()
                    alpha = torch.exp(-float(self.sde.theta) * t_anchor.float())
                    beta = (1.0 - alpha) * lam
                self.log("residual_anchor_lambda_mean", lam.mean(),
                         on_step=True, on_epoch=True, sync_dist=True, prog_bar=False)
                self.log("residual_anchor_lambda_min", lam.min(),
                         on_step=True, on_epoch=True, sync_dist=True, prog_bar=False)
                self.log("residual_anchor_lambda_max", lam.max(),
                         on_step=True, on_epoch=True, sync_dist=True, prog_bar=False)
                self.log("residual_anchor_beta_mean", beta.mean(),
                         on_step=True, on_epoch=True, sync_dist=True, prog_bar=False)
        else:
            if spk_ids is not None:
                step_out = self._step((X, Y_noisy, Y_pre, spk_ids), batch_idx, return_details=True)
            else:
                step_out = self._step((X, Y_noisy, Y_pre), batch_idx, return_details=True)

        raw_base_loss = step_out["base_loss"]
        raw_loss_total_backend = step_out["loss"]  # includes legacy contrastive if enabled
        # Legacy contrastive is orthogonal to regret-aware score reweighting.
        loss_old_contrastive = raw_loss_total_backend - raw_base_loss

        # Default: preserve the exact original score objective.
        base_loss = raw_base_loss
        regret_reweight_stats = {}
        if (
            self.frontend_nr_strategy == "score_reweight"
            and float(self.frontend_nr_reweight_strength) > 0.0
        ):
            if self.residual_refine_enable:
                # Frontend-only output is zero correction in residual coordinates.
                nr_target = r_target
                nr_frontend_reference = torch.zeros_like(r_target)
            else:
                nr_target = X
                nr_frontend_reference = Y_pre

            base_loss, regret_reweight_stats = self._compute_frontend_regret_score_reweighted_loss(
                target=nr_target,
                frontend_reference=nr_frontend_reference,
                step_out=step_out,
            )
            self.log(
                "train_loss_backend_raw", raw_base_loss,
                on_step=True, on_epoch=True, sync_dist=True, prog_bar=False,
            )
            self.log(
                "train_loss_backend_reweighted", base_loss,
                on_step=True, on_epoch=True, sync_dist=True, prog_bar=False,
            )
            for k, v in regret_reweight_stats.items():
                self.log(k, v, on_step=True, on_epoch=True, sync_dist=True, prog_bar=False)

        # Scheme A: keep the original DSM objective, but add a score-space
        # frontend anchor only where the current diffusion backend is locally
        # worse than the frontend. This is not output-level distillation.
        adaptive_guidance_stats = {}
        if (
            self.frontend_nr_strategy == "adaptive_score_guidance"
            and float(self.frontend_nr_guidance_strength) > 0.0
        ):
            if self.residual_refine_enable:
                nr_target = r_target
                nr_frontend_reference = torch.zeros_like(r_target)
            else:
                nr_target = X
                nr_frontend_reference = Y_pre

            guidance_loss, adaptive_guidance_stats = self._compute_frontend_adaptive_score_guidance_loss(
                target=nr_target,
                frontend_reference=nr_frontend_reference,
                step_out=step_out,
            )
            weighted_guidance = float(self.frontend_nr_guidance_strength) * guidance_loss
            base_loss = raw_base_loss + weighted_guidance.to(raw_base_loss.dtype)
            self.log(
                "train_loss_backend_raw", raw_base_loss,
                on_step=True, on_epoch=True, sync_dist=True, prog_bar=False,
            )
            self.log(
                "train_loss_frontend_score_guidance", guidance_loss,
                on_step=True, on_epoch=True, sync_dist=True, prog_bar=False,
            )
            self.log(
                "train_loss_frontend_score_guidance_weighted", weighted_guidance,
                on_step=True, on_epoch=True, sync_dist=True, prog_bar=False,
            )
            self.log(
                "train_loss_backend_guided", base_loss,
                on_step=True, on_epoch=True, sync_dist=True, prog_bar=False,
            )
            for k, v in adaptive_guidance_stats.items():
                self.log(k, v, on_step=True, on_epoch=True, sync_dist=True, prog_bar=False)

        loss_total_backend = base_loss + loss_old_contrastive
        x_hat_spec = step_out["x_hat_spec"]
        if self.residual_refine_enable and x_hat_spec is not None:
            # _step estimates in residual coordinates; downstream optional
            # clean-spectrum losses expect an enhanced clean-spectrum estimate.
            x_hat_spec = Y_pre.detach() + x_hat_spec
        loss = loss_total_backend
        for k, v in step_out.get("pathsim_stats", {}).items():
            self.log(k, v, on_step=True, on_epoch=True, sync_dist=True, prog_bar=False)

        # frontend 自己的辅助监督：时域 or 谱域
        if self.frontend_trainable and self.frontend_aux_loss != "none":
            loss_denoiser = self._compute_frontend_aux_loss(
                x_wav=x_wav,
                y_pre=y_pre,
                X=X,
                Y_pre=Y_pre,
            )
            a = float(self.weighting_denoiser_to_score)
            if not (0.0 <= a <= 1.0):
                raise ValueError(f"frontend_aux_weight should be in [0,1], got {a}")
            
            # 这里只对 frontend_aux 和 base_loss 做加权
            loss = a * loss_denoiser + (1.0 - a) * base_loss
             # 再把旧 contrastive 加回来
            loss = loss + loss_old_contrastive

            self.log("train_loss_denoiser", loss_denoiser, on_step=True, on_epoch=True, sync_dist=True, prog_bar=False)
        else:
            loss_denoiser = torch.zeros((), device=x_wav.device, dtype=x_wav.dtype)

        # Direct supervision for the inference-time residual initializer H.
        # Y_pre/Y_noisy were detached before H, so this branch updates H only.
        if self.residual_refine_enable:
            weighted_residual_init = self.residual_init_weight * loss_residual_init
            loss = loss + weighted_residual_init.to(loss.dtype)
            self.log("train_loss_residual_init", loss_residual_init,
                     on_step=True, on_epoch=True, sync_dist=True, prog_bar=False)
            self.log("train_loss_residual_init_weighted", weighted_residual_init,
                     on_step=True, on_epoch=True, sync_dist=True, prog_bar=False)
            self.log(
                "residual_init_weighted_to_backend_ratio",
                weighted_residual_init.detach() / (base_loss.detach().abs() + 1e-8),
                on_step=True, on_epoch=True, sync_dist=True, prog_bar=False,
            )
            for k, v in residual_init_stats.items():
                self.log(k, v, on_step=True, on_epoch=True, sync_dist=True, prog_bar=False)

        # ===== selector projection-oracle mask loss =====
        # This loss is only active for selector backbones that expose _last_selector_mask.
        # It should be kept small; it directly changes how much noisy evidence enters Y_hint.
        loss_selector_oracle = torch.zeros((), device=x_wav.device, dtype=torch.float32)
        if float(getattr(self, "selector_oracle_weight", 0.0)) > 0.0:
            selector_mask = getattr(self.dnn, "_last_selector_mask", None)
            residual_gates = getattr(self.dnn, "_last_residual_gates", None)
            noise_prior_gates = getattr(self.dnn, "_last_noise_prior_gates", None)
            has_residual_gates = residual_gates is not None and len(residual_gates) > 0
            has_noise_prior_gates = noise_prior_gates is not None and len(noise_prior_gates) > 0
            if selector_mask is not None:
                loss_selector_oracle, selector_oracle_stats = self._compute_selector_projection_oracle_loss(
                    X=X,
                    Y_noisy=Y_noisy,
                    Y_pre=Y_pre,
                    selector_mask=selector_mask,
                )
            elif has_residual_gates:
                loss_selector_oracle, selector_oracle_stats = self._compute_residual_regen_oracle_loss(
                    X=X,
                    Y_noisy=Y_noisy,
                    x_hat_spec=x_hat_spec,
                    residual_gates=residual_gates,
                )
            elif has_noise_prior_gates:
                loss_selector_oracle, selector_oracle_stats = self._compute_noise_prior_oracle_loss(
                    X=X,
                    Y_noisy=Y_noisy,
                    Y_pre=Y_pre,
                    noise_prior_gates=noise_prior_gates,
                )
            else:
                selector_oracle_stats = {}

            if selector_mask is not None or has_residual_gates or has_noise_prior_gates:
                weighted_selector_oracle = float(self.selector_oracle_weight) * loss_selector_oracle
                loss = loss + weighted_selector_oracle.to(loss.dtype)

                self.log("train_loss_selector_oracle", loss_selector_oracle,
                         on_step=True, on_epoch=True, sync_dist=True, prog_bar=False)
                self.log("train_loss_selector_oracle_weighted", weighted_selector_oracle,
                         on_step=True, on_epoch=True, sync_dist=True, prog_bar=False)
                for k, v in selector_oracle_stats.items():
                    self.log(k, v, on_step=True, on_epoch=True, sync_dist=True, prog_bar=False)
            else:
                # Useful when the command accidentally enables oracle loss without selector.
                self.log("train_loss_selector_oracle", loss_selector_oracle,
                         on_step=True, on_epoch=True, sync_dist=True, prog_bar=False)

        # ===== state-aware anchor calibration loss =====
        loss_anchor_calib = torch.zeros((), device=x_wav.device, dtype=torch.float32)
        if float(getattr(self, "anchor_calib_weight", 0.0)) > 0.0:
            loss_anchor_calib, anchor_stats = self._compute_anchor_calib_loss(
                X=X,
                Y_noisy=Y_noisy,
                Y_pre=Y_pre,
            )
            weighted_anchor_calib = float(self.anchor_calib_weight) * loss_anchor_calib
            loss = loss + weighted_anchor_calib.to(loss.dtype)
            self.log("train_loss_anchor_calib", loss_anchor_calib,
                     on_step=True, on_epoch=True, sync_dist=True, prog_bar=False)
            self.log("train_loss_anchor_calib_weighted", weighted_anchor_calib,
                     on_step=True, on_epoch=True, sync_dist=True, prog_bar=False)
            for k, v in anchor_stats.items():
                self.log(k, v, on_step=True, on_epoch=True, sync_dist=True, prog_bar=False)


        # ===== Anchor V2: fine-grained band-time x0 residual calibration =====
        loss_anchor_v2 = torch.zeros((), device=x_wav.device, dtype=torch.float32)
        if float(getattr(self, "anchor_v2_weight", 0.0)) > 0.0:
            loss_anchor_v2, anchor_v2_stats = self._compute_anchor_v2_loss(
                X=X,
                Y_noisy=Y_noisy,
                Y_pre=Y_pre,
            )
            weighted_anchor_v2 = float(self.anchor_v2_weight) * loss_anchor_v2
            loss = loss + weighted_anchor_v2.to(loss.dtype)
            self.log("train_loss_anchor_v2", loss_anchor_v2,
                     on_step=True, on_epoch=True, sync_dist=True, prog_bar=False)
            self.log("train_loss_anchor_v2_weighted", weighted_anchor_v2,
                     on_step=True, on_epoch=True, sync_dist=True, prog_bar=False)
            for k, v in anchor_v2_stats.items():
                self.log(k, v, on_step=True, on_epoch=True, sync_dist=True, prog_bar=False)

        # ===== RRSD: reliability routing =====
        # projection/kstep are oracle-distillation modes. future_grad instead
        # differentiates a one-step future re-score objective directly into Q_phi.
        # In all modes the frozen/base DSM model parameters remain untouched when
        # --rrsd_freeze_base is used.
        loss_rrsd = torch.zeros((), device=x_wav.device, dtype=torch.float32)
        if self.rrsd_router is not None and float(getattr(self, "rrsd_weight", 0.0)) > 0.0:
            # Full-speech RRSD: Oracle-distill how far the current diffusion x0
            # estimate should move toward the frontend enhanced spectrum Y_pre.
            loss_rrsd, rrsd_stats = self._compute_rrsd_loss(
                target=X,
                x0_front=Y_pre,
                step_out=step_out,
            )
            weighted_rrsd = float(self.rrsd_weight) * loss_rrsd
            loss = loss + weighted_rrsd.to(loss.dtype)
            self.log("train_loss_rrsd", loss_rrsd,
                     on_step=True, on_epoch=True, sync_dist=True, prog_bar=False)
            self.log("train_loss_rrsd_weighted", weighted_rrsd,
                     on_step=True, on_epoch=True, sync_dist=True, prog_bar=False)
            for k, v in rrsd_stats.items():
                self.log(k, v, on_step=True, on_epoch=True, sync_dist=True, prog_bar=False)

        # ===== final enhanced 的内容级 loss =====
        loss_content_final = torch.zeros((), device=x_wav.device, dtype=x_wav.dtype)
        if self.final_content_weight > 0.0 and x_hat_spec is not None and self.content_model is not None:
            x_hat_wav = self._spec_to_wave_batch(x_hat_spec)   # [B, T]
            loss_content_final = self._content_regression_loss_from_wavs(x_hat_wav, x_wav)
            weighted_content_final = self.final_content_weight * loss_content_final
            loss = loss + weighted_content_final
            self.log("train_loss_content_final_weighted", weighted_content_final, on_step=True, on_epoch=True, sync_dist=True)

        # ===== selector 的单独内容级辅助 loss =====
        loss_content_selector = torch.zeros((), device=x_wav.device, dtype=x_wav.dtype)
        selector_hint_spec = None
        if hasattr(self.dnn, "_last_selector_y_hint") and self.dnn._last_selector_y_hint is not None:
            # selector 的 y_hint 处在 c_in(t) 缩放坐标系里，拉回训练谱域
            t_for_this_step = step_out["t"]
            
            c_in_val = self._c_in(t_for_this_step)
            # 兼容 c_in='1' 这类返回 python float 的情况
            if not torch.is_tensor(c_in_val):
                c_in_val = torch.full_like(t_for_this_step, float(c_in_val))
            else:
                c_in_val = c_in_val.to(t_for_this_step.device, dtype=t_for_this_step.dtype)
            selector_hint_spec = self.dnn._last_selector_y_hint / (c_in_val[:, None, None, None] + 1e-12)

        if self.selector_content_weight > 0.0 and selector_hint_spec is not None and self.content_model is not None:
            selector_hint_wav = self._spec_to_wave_batch(selector_hint_spec)   # [B, T]
            loss_content_selector = self._content_regression_loss_from_wavs(selector_hint_wav, x_wav)
            weighted_content_selector = self.selector_content_weight * loss_content_selector
            loss = loss + weighted_content_selector
            self.log("train_loss_content_selector_weighted", weighted_content_selector, on_step=True, on_epoch=True, sync_dist=True)

        # ===== Legacy auxiliary frontend-relative no-regret loss (backend-only) =====
        # score_reweight/adaptive_score_guidance already modify the backend score
        # objective above and must not add this legacy auxiliary penalty.
        loss_frontend_nr = torch.zeros((), device=x_wav.device, dtype=torch.float32)
        if (
            self.frontend_nr_strategy == "auxiliary"
            and float(getattr(self, "frontend_nr_weight", 0.0)) > 0.0
        ):
            if self.residual_refine_enable:
                # Frontend-only output means zero correction in residual space.
                nr_target = r_target
                nr_frontend_reference = torch.zeros_like(r_target)
            else:
                nr_target = X
                nr_frontend_reference = Y_pre

            loss_frontend_nr, frontend_nr_stats = self._compute_frontend_no_regret_loss(
                target=nr_target,
                frontend_reference=nr_frontend_reference,
                step_out=step_out,
            )
            weighted_frontend_nr = float(self.frontend_nr_weight) * loss_frontend_nr
            loss = loss + weighted_frontend_nr.to(loss.dtype)
            self.log(
                "train_loss_frontend_nr", loss_frontend_nr,
                on_step=True, on_epoch=True, sync_dist=True, prog_bar=False,
            )
            self.log(
                "train_loss_frontend_nr_weighted", weighted_frontend_nr,
                on_step=True, on_epoch=True, sync_dist=True, prog_bar=False,
            )
            nr_to_backend = weighted_frontend_nr.detach() / loss_total_backend.detach().abs().clamp_min(
                float(self.frontend_nr_eps)
            )
            self.log(
                "frontend_nr_weighted_to_backend_ratio", nr_to_backend,
                on_step=True, on_epoch=True, sync_dist=True, prog_bar=False,
            )
            for k, v in frontend_nr_stats.items():
                self.log(k, v, on_step=True, on_epoch=True, sync_dist=True, prog_bar=False)

        self.log("train_loss_backend", loss_total_backend, on_step=True, on_epoch=True, sync_dist=True, prog_bar=False)
        if self.final_content_weight > 0.0:
            self.log("train_loss_content_final", loss_content_final, on_step=True, on_epoch=True, sync_dist=True, prog_bar=False)
        if self.selector_content_weight > 0.0:
            self.log("train_loss_content_selector", loss_content_selector, on_step=True, on_epoch=True, sync_dist=True, prog_bar=False)
        
        self.log("train_loss", loss, on_step=True, on_epoch=True, sync_dist=True, prog_bar=True)
        return loss
    
    def _get_valid_file_lists(self, num_eval: int):
        """兼容 base_dir(WavePairs) 和 wav.scp(WavePairsScp) 两种 valid_set."""
        vs = self.data_module.valid_set

        # base_dir: WavePairs 有 clean_files/noisy_files
        if hasattr(vs, "clean_files") and hasattr(vs, "noisy_files"):
            clean_all = vs.clean_files[:num_eval]
            noisy_all = vs.noisy_files[:num_eval]
            return clean_all, noisy_all

        # wav.scp: WavePairsScp 有 uttids/noisy_files/_resolve_clean_path
        if hasattr(vs, "uttids") and hasattr(vs, "noisy_files") and hasattr(vs, "_resolve_clean_path"):
            noisy_all = vs.noisy_files[:num_eval]
            uttids = vs.uttids[:num_eval]
            clean_all = [vs._resolve_clean_path(u, p) for u, p in zip(uttids, noisy_all)]
            return clean_all, noisy_all

        raise RuntimeError(
            "valid_set does not expose file lists. "
            "Expected WavePairs(clean_files/noisy_files) or WavePairsScp(uttids/noisy_files/_resolve_clean_path)."
        )

    def _cuda_mem(self, prefix=""):
        if torch.cuda.is_available():
            dev = torch.cuda.current_device()
            alloc = torch.cuda.memory_allocated(dev) / 1024**2
            reserv = torch.cuda.memory_reserved(dev) / 1024**2
            peak = torch.cuda.max_memory_allocated(dev) / 1024**2
            print(f"[rank0][mem] {prefix} alloc={alloc:.1f}MB reserved={reserv:.1f}MB peak={peak:.1f}MB", flush=True)
    def _eval_metrics_on_files_single_rank(self, num_eval):
        """只在 rank 0 上做整套文件级评测：PESQ / SI-SDR / ESTOI。
        不做任何 DDP all_reduce，不参与 collectives。
        """
        num_eval = int(getattr(self, "num_eval_files", 0) or 0)
        if num_eval <= 0:
            return None

        clean_files, noisy_files = self._get_valid_file_lists(num_eval)

        local_pesq_sum = 0.0
        local_si_sdr_sum = 0.0
        local_estoi_sum = 0.0
        local_count = 0.0

        with torch.no_grad():
            for idx,(clean_file, noisy_file) in enumerate(zip(clean_files, noisy_files)):
                try:
                    if torch.cuda.is_available():
                        torch.cuda.reset_peak_memory_stats()
                        self._cuda_mem(f"before file {idx}")
                    x, sr_x = load(clean_file)   # [C,T]
                    y, sr_y = load(noisy_file)

                    x = x[:1, :]  # mono
                    y = y[:1, :]

                    x = x.squeeze(0).cpu().numpy()
                    y = y.squeeze(0).cpu().numpy()

                    # 采样率不一致时，先统一
                    if sr_x != sr_y:
                        min_sr = min(sr_x, sr_y)
                        x = resample(x, orig_sr=sr_x, target_sr=min_sr)
                        y = resample(y, orig_sr=sr_y, target_sr=min_sr)
                        sr_x = sr_y = min_sr

                    # 统一长度
                    L = min(len(x), len(y))
                    if L < 1600:
                        continue
                    x = x[:L]
                    y = y[:L]

                    # Joint 推理：noisy -> SEMamba -> y_pre -> SGMSE enhance
                    y_t = torch.from_numpy(y).float().unsqueeze(0).unsqueeze(0).to(self.device)  # [1,1,T]
                    norm = y_t.abs().amax(dim=-1, keepdim=True).clamp_min(1e-12)
                    if self.frontend_name == "storm" and hasattr(self.frontend, "forward_spec"):
                        Y_pre_n = self.frontend.forward_spec(y_t / norm)
                        y_pre = self.frontend.spec_to_waveform(Y_pre_n, length=y_t.size(-1)) * norm
                        y_pre = _match_length(y_pre, y_t)
                    else:
                        y_pre = self._frontend_forward(y_t / norm) * norm
                        y_pre = _match_length(y_pre, y_t)
                    if self.train_type == "mode2":
                        y_in = y_pre.squeeze(0)   # [1,T]
                        y_cond = y_t.squeeze(0)   # [1,T]
                    else:  # mode3
                        y_in = y_t.squeeze(0)     # [1,T]
                        y_cond = y_pre.squeeze(0) # [1,T]

                    x_hat = self.enhance(
                        y_in,
                        y_cond=y_cond,
                        N=self.sde.N,
                        anchor_v2=bool(getattr(self, "anchor_v2_validate", False)),
                        rrsd=bool(getattr(self, "rrsd_validate", False)),
                        rrsd_scale=1.0,
                        anchor_v2_single_shot=bool(getattr(self, "anchor_v2_validate_single_shot", False)),
                        anchor_v2_single_shot_ratio=float(getattr(self, "anchor_v2_single_shot_ratio", 0.5)),
                    )
                    self._cuda_mem(f"after enhance {idx}")
                    if isinstance(x_hat, torch.Tensor):
                        x_hat = x_hat.detach().float().cpu().numpy()
                    self._cuda_mem(f"after x_hat to cpu {idx}")
                    # 统一到 16k，PESQ(wb) 需要 16k
                    x_16k = resample(x, orig_sr=sr_x, target_sr=16000) if sr_x != 16000 else x
                    x_hat_16k = resample(x_hat, orig_sr=sr_y, target_sr=16000) if sr_y != 16000 else x_hat

                    # 对齐长度
                    L2 = min(len(x_16k), len(x_hat_16k))
                    if L2 < 1600:
                        continue
                    x_16k = np.clip(x_16k[:L2], -1.0, 1.0)
                    x_hat_16k = np.clip(x_hat_16k[:L2], -1.0, 1.0)

                    local_pesq_sum += float(pesq(16000, x_16k, x_hat_16k, "wb"))
                    local_si_sdr_sum += float(si_sdr(x_16k, x_hat_16k))
                    local_estoi_sum += float(stoi(x_16k, x_hat_16k, 16000, extended=True))
                    local_count += 1.0

                except Exception as e:
                    print(
                        f"[rank0][eval_metrics] failed on clean={clean_file}, noisy={noisy_file}, err={repr(e)}",
                        flush=True,
                    )
                    continue

        if local_count <= 0:
            print("[rank0][eval_metrics] no valid files were evaluated.", flush=True)
            return None

        g_pesq = local_pesq_sum / local_count
        g_si = local_si_sdr_sum / local_count
        g_estoi = local_estoi_sum / local_count
        return g_pesq, g_si, g_estoi
    
    def on_validation_epoch_end(self):
        """验证 epoch 结束后，只在 rank 0 做整套文件级评测。
        其它 rank 在 barrier 等待，避免下一轮训练提前开始造成 DDP 失同步。
        每隔 10 个 epoch 才评测一次。
        """
        # Lightning 的 sanity check 阶段不跑这套慢评测
        if getattr(self.trainer, "sanity_checking", False):
            return
        
        # 每隔 10 个 epoch 才跑一次
        # current_epoch 从 0 开始，所以会在 9, 19, 29, ... 跑
        if (self.current_epoch + 1) % 10 != 0:
            return
        
        num_eval = int(getattr(self, "num_eval_files", 0) or 0)
        if num_eval <= 0:
            print(f"num_evel=0, skip file-level validation metrics")
            return None
        
        is_global_zero = bool(getattr(self.trainer, "is_global_zero", True))

        metrics = None
        if is_global_zero:
            print(f"[rank0] start file-level validation metrics at epoch={self.current_epoch}", flush=True)
            metrics = self._eval_metrics_on_files_single_rank(num_eval=num_eval)
            print(f"[rank0] finish file-level validation metrics at epoch={self.current_epoch}", flush=True)

            if metrics is not None:
                g_pesq, g_si, g_estoi = metrics
                self.log("val_pesq", g_pesq, prog_bar=True, rank_zero_only=True)
                self.log("val_si_sdr", g_si, prog_bar=True, rank_zero_only=True)
                self.log("val_estoi", g_estoi, prog_bar=False, rank_zero_only=True)

        # 非 rank0 这里等一下，避免 rank0 还在做 enhance/PESQ，
        # 其它 rank 已经进入下一轮训练，导致 collective 失同步
        # gemini说Lightning 会自动处理 epoch 级别的同步，所以删除
        # if is_distributed:
        #     dist.barrier()

    def validation_step(self, batch, batch_idx):
        # 不要在 validation_step 里做全文件评测
        # 文件级评测已经挪到 on_validation_epoch_end
        if len(batch) == 2:
            x_wav, y_wav = batch
            spk_ids = None
        else:
            x_wav, y_wav, spk_ids = batch

        # 验证阶段不要重新开 grad，先用纯 no_grad 更稳
        with torch.no_grad():
            X = self._sgmse_stft(x_wav)
            Y_noisy = self._sgmse_stft(y_wav)

            if self.frontend_name == "storm" and hasattr(self.frontend, "forward_spec"):
                # Reuse Y_noisy to avoid second STFT + spectral compression.
                # Frontend and data-module spectral configs must match.
                Y_pre = self.frontend.forward_transformed_spec(Y_noisy)
            else:
                y_pre = self._frontend_forward(y_wav)
                y_pre = _match_length(y_pre, y_wav)
                Y_pre = self._sgmse_stft(y_pre)

            if self.residual_refine_enable:
                r_target, r_obs, r_det, r_center = self._build_predictive_residual_state(
                    X=X, Y_noisy=Y_noisy, Y_pre=Y_pre
                )
                if spk_ids is not None:
                    residual_step_out = self._step(
                        (r_target, r_obs, r_center, spk_ids), batch_idx, return_details=True
                    )
                else:
                    residual_step_out = self._step(
                        (r_target, r_obs, r_center), batch_idx, return_details=True
                    )
                loss_score = residual_step_out["loss"]
                loss_init, init_stats = self._compute_residual_initializer_loss(
                    r_target=r_target, r_obs=r_obs, r_det=r_det
                )
                if hasattr(self.sde, "residual_anchor_lambda"):
                    t_anchor = residual_step_out["t"].detach()
                    lam = self.sde.residual_anchor_lambda(t_anchor).float()
                    alpha = torch.exp(-float(self.sde.theta) * t_anchor.float())
                    beta = (1.0 - alpha) * lam
                    self.log("valid_residual_anchor_lambda_mean", lam.mean(),
                             on_step=False, on_epoch=True, sync_dist=True, prog_bar=False)
                    self.log("valid_residual_anchor_beta_mean", beta.mean(),
                             on_step=False, on_epoch=True, sync_dist=True, prog_bar=False)
                loss = loss_score + self.residual_init_weight * loss_init
                self.log("valid_loss_residual_score", loss_score,
                         on_step=False, on_epoch=True, sync_dist=True, prog_bar=False)
                self.log("valid_loss_residual_init", loss_init,
                         on_step=False, on_epoch=True, sync_dist=True, prog_bar=False)
                for k, v in init_stats.items():
                    self.log("valid_" + k, v, on_step=False, on_epoch=True,
                             sync_dist=True, prog_bar=False)
            else:
                if spk_ids is not None:
                    loss = self._step((X, Y_noisy, Y_pre, spk_ids), batch_idx)
                else:
                    loss = self._step((X, Y_noisy, Y_pre), batch_idx)
        self.log("valid_loss", loss, on_step=False, on_epoch=True, sync_dist=True, prog_bar=True)

        return loss


# 向后兼容旧类名
JointSemambaSgmseModel = JointFrontendSgmseModel
