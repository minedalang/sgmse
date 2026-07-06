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
from sgmse.util.other import si_sdr

from sgmse.model import ScoreModel
from sgmse.sdes import SBVESDE
from sgmse.frontends import build_frontend

from contextlib import contextmanager
import copy
import os

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


    def configure_optimizers(self):
        params = [p for p in self.parameters() if p.requires_grad]
        optimizer = torch.optim.Adam(params, lr=self.lr)
        return optimizer
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
            Y_pre = self.frontend.forward_spec(y_wav)
            y_pre = self.frontend.spec_to_waveform(Y_pre, length=y_wav.size(-1))
            y_pre = _match_length(y_pre, y_wav)
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

        # 组 batch 给 ScoreModel._step
        if spk_ids is not None:
            step_out  = self._step((X, Y_noisy, Y_pre, spk_ids), batch_idx, return_details=True)
        else:
            step_out  = self._step((X, Y_noisy, Y_pre), batch_idx, return_details=True)
        base_loss = step_out["base_loss"]
        loss_total_backend = step_out["loss"] # 包含旧 contrastive
        x_hat_spec = step_out["x_hat_spec"]
        loss = loss_total_backend
        # 旧 contrastive 单独拆出来
        loss_old_contrastive = loss_total_backend - base_loss

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

                    x_hat = self.enhance(y_in, y_cond=y_cond, N=self.sde.N)
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
                Y_pre = self.frontend.forward_spec(y_wav)
            else:
                y_pre = self._frontend_forward(y_wav)
                y_pre = _match_length(y_pre, y_wav)
                Y_pre = self._sgmse_stft(y_pre)

            if spk_ids is not None:
                loss = self._step((X, Y_noisy, Y_pre, spk_ids), batch_idx)
            else:
                loss = self._step((X, Y_noisy, Y_pre), batch_idx)
        self.log("valid_loss", loss, on_step=False, on_epoch=True, sync_dist=True, prog_bar=True)

        return loss


# 向后兼容旧类名
JointSemambaSgmseModel = JointFrontendSgmseModel
