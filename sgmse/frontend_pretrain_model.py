from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F
import pytorch_lightning as pl

from sgmse.frontends import build_frontend, match_length


class FrontendPretrainModel(pl.LightningModule):
    """单独预训练前端模型。

    输入：noisy waveform
    目标：clean waveform
    损失：默认 time-domain L1
    """

    @staticmethod
    def add_argparse_args(parser):
        parser.add_argument("--lr", type=float, default=1e-4)
        parser.add_argument("--frontend", type=str, choices=("semamba", "storm", "tfgridnet"), default="semamba")
        parser.add_argument("--frontend_root", type=str, default=None)
        parser.add_argument("--frontend_config", type=str, default=None)
        parser.add_argument("--frontend_ckpt", type=str, default=None)
        parser.add_argument("--frontend_module", type=str, default=None)
        parser.add_argument("--frontend_class", type=str, default=None)
        parser.add_argument("--frontend_state_dict_key", type=str, default=None)
        parser.add_argument("--pretrain_loss", type=str, choices=("l1", "mse"), default="l1")
        parser.add_argument("--semamba_root", type=str, default=None, help="兼容旧参数")
        parser.add_argument("--semamba_config", type=str, default=None, help="兼容旧参数")
        parser.add_argument("--semamba_ckpt", type=str, default=None, help="兼容旧参数")
        return parser

    def __init__(
        self,
        lr: float = 1e-4,
        frontend: str = "semamba",
        frontend_root: Optional[str] = None,
        frontend_config: Optional[str] = None,
        frontend_ckpt: Optional[str] = None,
        frontend_module: Optional[str] = None,
        frontend_class: Optional[str] = None,
        frontend_state_dict_key: Optional[str] = None,
        pretrain_loss: str = "l1",
        semamba_root: Optional[str] = None,
        semamba_config: Optional[str] = None,
        semamba_ckpt: Optional[str] = None,
        data_module_cls=None,
        **kwargs,
    ):
        super().__init__()
        if frontend == "semamba":
            frontend_root = frontend_root or semamba_root
            frontend_config = frontend_config or semamba_config
            frontend_ckpt = frontend_ckpt or semamba_ckpt

        self.lr = float(lr)
        self.pretrain_loss = str(pretrain_loss)
        self.frontend_name = str(frontend)
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
        self.data_module = data_module_cls(**kwargs, gpu=kwargs.get("gpus", 0) > 0)
        self.save_hyperparameters(ignore=["data_module_cls"])

    def configure_optimizers(self):
        return torch.optim.Adam(self.frontend.parameters(), lr=self.lr)

    def _compute_loss(self, clean_wav: torch.Tensor, pred_wav: torch.Tensor) -> torch.Tensor:
        if self.pretrain_loss == "mse":
            return F.mse_loss(pred_wav, clean_wav)
        return F.l1_loss(pred_wav, clean_wav)

    def _unpack_batch(self, batch):
        if len(batch) == 2:
            clean_wav, noisy_wav = batch
        else:
            clean_wav, noisy_wav, *_ = batch
        return clean_wav, noisy_wav

    def training_step(self, batch, batch_idx):
        clean_wav, noisy_wav = self._unpack_batch(batch)
        pred_wav = self.frontend(noisy_wav)
        pred_wav = match_length(pred_wav, clean_wav)
        loss = self._compute_loss(clean_wav, pred_wav)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        clean_wav, noisy_wav = self._unpack_batch(batch)
        with torch.no_grad():
            pred_wav = self.frontend(noisy_wav)
            pred_wav = match_length(pred_wav, clean_wav)
            loss = self._compute_loss(clean_wav, pred_wav)
        self.log("valid_loss", loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        return loss
