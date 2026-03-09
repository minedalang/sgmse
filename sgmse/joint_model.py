"""sgmse.joint_model

联合训练 LightningModule：SEMamba(前端) + ScoreModel/SGMSE(后端)。

核心思想：
- batch 提供 clean/noisy 波形
- noisy 波形先过 SEMamba 得到 y_pre (可微)
- SGMSE 的 score network 接收 (y=noisy, y_cond=y_pre) 或 (y=y_pre, y_cond=noisy)
- 使用 SGMSE 标准训练目标（score-matching / denoiser loss）
- loss 反向传播同时更新 SEMamba 与 SGMSE

实现细节：
- 由于 SEMamba 仓库的 generator.py 使用相对导入（.mamba_block 等），
  这里用“虚拟包”方式从 semamba_root/models 动态导入，避免强制改目录名。
"""

from __future__ import annotations

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
from torchaudio import load
from librosa import resample
from pesq import pesq
from pystoi import stoi
from sgmse.util.other import si_sdr

from sgmse.model import ScoreModel


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


class JointSemambaSgmseModel(ScoreModel):
    strict_loading = False
    @staticmethod
    def add_argparse_args(parser):
        # 继承 ScoreModel 的参数
        ScoreModel.add_argparse_args(parser)

        # 增加的 SEMamba相关参数
        parser.add_argument("--semamba_root", type=str, required=True, help="Path to Semamba-2-10 repo root")
        parser.add_argument("--semamba_config", type=str, required=True, help="Path to Semamba config.yaml")
        parser.add_argument("--semamba_ckpt", type=str, default=None, help="Optional SEMamba checkpoint .pth")
        parser.add_argument("--semamba_trainable", action="store_true", help="Enable joint training for SEMamba")
        parser.add_argument("--semamba_lr_mult", type=float, default=0.1, help="SEMamba lr multiplier relative to --lr")
        parser.add_argument(
            "--semamba_aux_loss",
            type=str,
            choices=("none", "time_l1"),
            default="none",
            help="Optional SEMamba auxiliary supervision (stabilizes joint training)",
        )
        parser.add_argument("--semamba_aux_weight", type=float, default=0.1)
        return parser

    def __init__(
        self,
        backbone: str,
        sde: str,
        train_type: str = "mode3",
        semamba_root: str = "",
        semamba_config: str = "",
        semamba_ckpt: Optional[str] = None,
        semamba_trainable: bool = False,
        semamba_lr_mult: float = 0.1,
        semamba_aux_loss: str = "none",
        semamba_aux_weight: float = 0.1,
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

        self.semamba_lr_mult = semamba_lr_mult
        self.semamba_aux_loss = semamba_aux_loss
        self.semamba_aux_weight = semamba_aux_weight

        self.semamba, self.semamba_cfg, self._sem_stft, self._sem_istft = _load_semamba(
            semamba_root=semamba_root,
            config_path=semamba_config,
            ckpt_path=semamba_ckpt,
        )

        # STFT cfg from SEMamba
        stft_cfg = self.semamba_cfg.get("stft_cfg", {})
        model_cfg = self.semamba_cfg.get("model_cfg", {})
        self.sem_n_fft = int(stft_cfg.get("n_fft", 400))
        self.sem_hop = int(stft_cfg.get("hop_size", 100))
        self.sem_win = int(stft_cfg.get("win_size", 400))
        self.sem_cf = float(model_cfg.get("compress_factor", 1.0))

        self.semamba_trainable = bool(semamba_trainable)
        if not self.semamba_trainable:
            for p in self.semamba.parameters():
                p.requires_grad = False
            self.semamba.eval()

    def configure_optimizers(self):
        # 给 SEMamba 单独一个 lr（可选）
        base_params = []
        sem_params = []
        for n, p in self.named_parameters():
            if not p.requires_grad:
                continue
            if n.startswith("semamba."):
                sem_params.append(p)
            else:
                base_params.append(p)

        if self.semamba_trainable and sem_params:
            optimizer = torch.optim.Adam(
                [
                    {"params": base_params, "lr": self.lr},
                    {"params": sem_params, "lr": self.lr * float(self.semamba_lr_mult)},
                ]
            )
        else:
            optimizer = torch.optim.Adam(base_params, lr=self.lr)
        return optimizer

    def _semamba_forward(self, y_noisy: torch.Tensor) -> torch.Tensor:
        """y_noisy: [B,1,T] -> y_pre: [B,1,T]"""
        y = y_noisy.squeeze(1)  # [B, T]
        noisy_mag, noisy_pha, _ = self._sem_stft(
            y,
            n_fft=self.sem_n_fft,
            hop_size=self.sem_hop,
            win_size=self.sem_win,
            compress_factor=self.sem_cf,
            center=True,
            addeps=False,
        )
        den_mag, den_pha, _ = self.semamba(noisy_mag, noisy_pha)
        y_pre = self._sem_istft(
            den_mag,
            den_pha,
            n_fft=self.sem_n_fft,
            hop_size=self.sem_hop,
            win_size=self.sem_win,
            compress_factor=self.sem_cf,
            center=True,
        )
        return y_pre.unsqueeze(1)  # [B,1,T']

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

    def training_step(self, batch, batch_idx):
        # batch: (x_wav, y_wav, spk_id)
        if len(batch) == 2:
            x_wav, y_wav = batch
            spk_ids = None
        else:
            x_wav, y_wav, spk_ids = batch

        # ======== [DEBUG] 打印输入统计 ========
        # self._print_tensor_stats("x_wav(clean)", x_wav, batch_idx)
        # self._print_tensor_stats("y_wav(noisy)", y_wav, batch_idx)

        # SEMamba 前向：noisy -> y_pre
        y_pre = self._semamba_forward(y_wav)
        y_pre = _match_length(y_pre, y_wav)
        
        # ======== [DEBUG] 打印 SEMamba 输出 ========
        # self._print_tensor_stats("y_pre(SEMamba)", y_pre, batch_idx)


        # 频谱域：SGMSE 使用自己的 STFT + spec_fwd 得到时频谱
        X = self._sgmse_stft(x_wav)
        Y_noisy = self._sgmse_stft(y_wav)
        Y_pre = self._sgmse_stft(y_pre)

        # ======== [DEBUG] 打印 STFT 频谱统计 ========
        # self._print_tensor_stats("X(clean_spec)", X, batch_idx)
        # self._print_tensor_stats("Y_noisy(noisy_spec)", Y_noisy, batch_idx)
        # self._print_tensor_stats("Y_pre(semamba_spec)", Y_pre, batch_idx)

        # 组 batch 给 ScoreModel._step
        if spk_ids is not None:
            loss_sgmse = self._step((X, Y_noisy, Y_pre, spk_ids), batch_idx)
        else:
            loss_sgmse = self._step((X, Y_noisy, Y_pre), batch_idx)

        loss = loss_sgmse


        # 可选：给 SEMamba 一个简单的时间域监督，稳定 joint training
        if self.semamba_trainable and self.semamba_aux_loss == "time_l1":
            aux = torch.mean(torch.abs(y_pre - x_wav))
            self.log("semamba_aux", aux, on_step=True, on_epoch=True, sync_dist=True, prog_bar=False)
            loss = loss + self.semamba_aux_weight * aux

        # ======== [DEBUG] 打印梯度范数 ========
        # if batch_idx % 100 == 0:
        #     total_grad_norm = 0.0
        #     for n, p in self.named_parameters():
        #         if p.grad is not None:
        #             grad_norm = p.grad.norm().item()
        #             total_grad_norm += grad_norm ** 2
        #     total_grad_norm = total_grad_norm ** 0.5
        #     print(f"[DEBUG:{batch_idx}] total_grad_norm={total_grad_norm:.6f}")

        self.log("train_loss", loss, on_step=True, on_epoch=True, sync_dist=True, prog_bar=True)
        self.log("train_loss_sgmse", loss_sgmse, on_step=True, on_epoch=True, sync_dist=True, prog_bar=False)
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


    def _eval_metrics_on_files(self):
        """每个 epoch 的第一个 val batch 调用：全文件推理并算 PESQ/SI-SDR/ESTOI（DDP汇总）。"""
        num_eval = int(getattr(self, "num_eval_files", 0) or 0)
        if num_eval <= 0:
            return

        # DDP 信息
        if dist.is_available() and dist.is_initialized():
            world_size = dist.get_world_size()
            rank = dist.get_rank()
        else:
            world_size, rank = 1, 0

        per_gpu = int(math.ceil(num_eval / world_size))
        start = rank * per_gpu
        end = min((rank + 1) * per_gpu, num_eval)

        clean_all, noisy_all = self._get_valid_file_lists(num_eval)
        clean_files = clean_all[start:end]
        noisy_files = noisy_all[start:end]

        local_pesq_sum = 0.0
        local_si_sdr_sum = 0.0
        local_estoi_sum = 0.0
        local_count = 0.0

        # 推理（注意：self.enhance 会走扩散采样，很慢，所以 num_eval_files 建议设小一点，比如 5~20）
        with torch.no_grad():
            for clean_file, noisy_file in zip(clean_files, noisy_files):
                try:
                    x, sr_x = load(clean_file)   # [C,T]
                    y, sr_y = load(noisy_file)
                    x = x[:1, :]  # mono
                    y = y[:1, :]
                    x = x.squeeze(0).cpu().numpy()
                    y = y.squeeze(0).cpu().numpy()

                    # （可选）确保采样率一致；你训练一般是 16k，严格对齐更好
                    if sr_x != sr_y:
                        min_sr = min(sr_x, sr_y)
                        x = resample(x, orig_sr=sr_x, target_sr=min_sr)
                        y = resample(y, orig_sr=sr_y, target_sr=min_sr)
                        sr_x = sr_y = min_sr

                    # 统一长度
                    L = min(len(x), len(y))
                    if L < 1600:  # 太短就跳过，避免 PESQ 报错（可按需调）
                        continue
                    x = x[:L]
                    y = y[:L]

                    # === Joint 推理：noisy -> SEMamba -> y_pre -> SGMSE enhance ===
                    y_t = torch.from_numpy(y).float().unsqueeze(0).unsqueeze(0).to(self.device)  # [1,1,T]
                    y_pre = self._semamba_forward(y_t)            # [1,1,T']
                    y_pre = _match_length(y_pre, y_t)             # [1,1,T]

                    if self.train_type == "mode2":
                        y_in = y_pre.squeeze(0)   # [1,T]
                        y_cond = y_t.squeeze(0)   # [1,T]
                    else:  # mode3
                        y_in = y_t.squeeze(0)     # [1,T]
                        y_cond = y_pre.squeeze(0) # [1,T]

                    # enhance() 在你的 ScoreModel 版本里返回 numpy.ndarray (T,)
                    x_hat = self.enhance(y_in, y_cond=y_cond, N=self.sde.N)

                    # === 评测统一到 16k（PESQ wb 需要 16k） ===
                    x_16k = resample(x, orig_sr=sr_x, target_sr=16000) if sr_x != 16000 else x
                    x_hat_16k = resample(x_hat, orig_sr=sr_y, target_sr=16000) if sr_y != 16000 else x_hat

                    # 对齐长度 + clip，避免 pesq/si_sdr 出错
                    L2 = min(len(x_16k), len(x_hat_16k))
                    if L2 < 1600:
                        continue
                    x_16k = np.clip(x_16k[:L2], -1.0, 1.0)
                    x_hat_16k = np.clip(x_hat_16k[:L2], -1.0, 1.0)

                    local_pesq_sum += float(pesq(16000, x_16k, x_hat_16k, "wb"))
                    local_si_sdr_sum += float(si_sdr(x_16k, x_hat_16k))
                    local_estoi_sum += float(stoi(x_16k, x_hat_16k, 16000, extended=True))
                    local_count += 1.0

                except Exception:
                    # 单个文件评测失败就跳过，不让训练崩
                    continue

        # DDP 汇总
        t = torch.tensor(
            [local_pesq_sum, local_si_sdr_sum, local_estoi_sum, local_count],
            device=self.device,
            dtype=torch.float64,
        )
        if dist.is_available() and dist.is_initialized() and world_size > 1:
            dist.all_reduce(t, op=dist.ReduceOp.SUM)

        g_count = max(float(t[3].item()), 1.0)
        g_pesq = (t[0] / g_count).item()
        g_si = (t[1] / g_count).item()
        g_estoi = (t[2] / g_count).item()

        # 我们已经手动 all_reduce 了，所以 sync_dist=False
        self.log("val_pesq", g_pesq, on_step=False, on_epoch=True, prog_bar=True, sync_dist=False)
        self.log("val_si_sdr", g_si, on_step=False, on_epoch=True, prog_bar=True, sync_dist=False)
        self.log("val_estoi", g_estoi, on_step=False, on_epoch=True, prog_bar=False, sync_dist=False)
    def validation_step(self, batch, batch_idx):
        # 只在每个验证 epoch 的第一个 batch 做一次全文件评测
        if batch_idx == 0:
            self._eval_metrics_on_files()

        # 原来的 valid_loss 仍保留（扩散训练目标）
        if len(batch) == 2:
            x_wav, y_wav = batch
            spk_ids = None
        else:
            x_wav, y_wav, spk_ids = batch

        with torch.set_grad_enabled(self.semamba_trainable):
            y_pre = self._semamba_forward(y_wav)
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
