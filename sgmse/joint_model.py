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
            return torch.mean(
                    0.5 * torch.sum(loss_tmp.reshape(loss_tmp.shape[0], -1), dim=-1)
                )
        elif loss_type == "time_mse":
            loss_tmp =  (y_pre, x_wav) **2
            return torch.mean(
                    0.5 * torch.sum(loss_tmp.reshape(loss_tmp.shape[0], -1), dim=-1)
                )
        elif loss_type == "spec_mae":
            # 直接在训练域复谱上做 L1（复数用 abs 表示复距离）
            loss_tmp = torch.abs(Y_pre - X)
            return torch.mean(
                    0.5 * torch.sum(loss_tmp.reshape(loss_tmp.shape[0], -1), dim=-1)
                )

        elif loss_type == "spec_mse":  #这个似乎是storm默认loss
            # 直接在训练域复谱上做 L2
            loss_tmp = torch.abs(Y_pre - X) ** 2
            return torch.mean(
                    0.5 * torch.sum(loss_tmp.reshape(loss_tmp.shape[0], -1), dim=-1)
                )
        elif loss_type == "spec_cplx_mag":
            # 复谱差 + 幅度差，通常比纯 time_mse 更像 storm 那种约束思路
            loss_tmp = torch.abs(Y_pre - X) ** 2
            loss_cplx = torch.mean(
                    0.5 * torch.sum(loss_tmp.reshape(loss_tmp.shape[0], -1), dim=-1)
                )
            loss_tmp = (torch.abs(Y_pre) - torch.abs(X)) ** 2
            loss_mag = torch.mean(
                    0.5 * torch.sum(loss_tmp.reshape(loss_tmp.shape[0], -1), dim=-1)
                )
            return loss_cplx + 0.5 * loss_mag

        elif loss_type == "none":
            return torch.zeros((), device=x_wav.device, dtype=x_wav.dtype)

        else:
            raise ValueError(f"Unsupported frontend_aux_loss: {loss_type}")
        
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

        # SEMamba 前向：noisy -> y_pre
        y_pre = self._frontend_forward(y_wav)
        y_pre = _match_length(y_pre, y_wav)
        # ======== [DEBUG] 打印 SEMamba 输出 ========
        # self._print_tensor_stats("y_pre(SEMamba)", y_pre, batch_idx)


        # 频谱域：SGMSE 使用自己的 STFT + spec_fwd 得到时频谱
        X = self._sgmse_stft(x_wav)
        Y_noisy = self._sgmse_stft(y_wav)
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
                    y_pre = self._frontend_forward(y_t / norm) * norm   # [1,1,T']
                    y_pre = _match_length(y_pre, y_t)    # [1,1,T]
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
            y_pre = self._frontend_forward(y_wav)
            y_pre = _match_length(y_pre, y_wav)

            X = self._sgmse_stft(x_wav)
            Y_noisy = self._sgmse_stft(y_wav)
            Y_pre = self._sgmse_stft(y_pre)

            if spk_ids is not None:
                loss = self._step((X, Y_noisy, Y_pre, spk_ids), batch_idx)
            else:
                loss = self._step((X, Y_noisy, Y_pre), batch_idx)
        self.log("valid_loss", loss, on_step=False, on_epoch=True, sync_dist=True, prog_bar=True)

        return loss


# 向后兼容旧类名
JointSemambaSgmseModel = JointFrontendSgmseModel
