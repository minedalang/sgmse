from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path
from typing import Any, Dict, Optional

import torch

from .base import BaseFrontend, ensure_mono_waveform, match_length, strip_state_dict_prefixes
from .registry import FrontendRegistry


def _load_yaml_or_json(path: Optional[str]) -> Dict[str, Any]:
    if not path:
        return {}
    path_obj = Path(path)
    suffix = path_obj.suffix.lower()
    if suffix in (".yml", ".yaml"):
        import yaml
        with open(path_obj, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    if suffix == ".json":
        with open(path_obj, "r", encoding="utf-8") as f:
            return json.load(f)
    raise ValueError(f"Unsupported config suffix: {suffix}")


def _guess_state_dict(ckpt: Any, state_dict_key: Optional[str] = None) -> Dict[str, Any]:
    if isinstance(ckpt, dict):
        if state_dict_key and state_dict_key in ckpt and isinstance(ckpt[state_dict_key], dict):
            return ckpt[state_dict_key]
        for key in ("state_dict", "model", "generator", "net"):
            if key in ckpt and isinstance(ckpt[key], dict):
                return ckpt[key]
    if isinstance(ckpt, dict):
        return ckpt
    raise RuntimeError("Cannot parse checkpoint into a state_dict")


@FrontendRegistry.register("tfgridnet")
class TFGridNetFrontend(BaseFrontend):
    """ESPnet TF-GridNet 前端适配器。

    官方 `TFGridNet.forward()` 接收 `input` 和 `ilens`，并返回
    `(enhanced_list, ilens, additional)`；对单通道增强我们固定取第一个源。
    为了做 speech enhancement，建议把 `n_srcs` 设为 1。
    """

    @staticmethod
    def add_argparse_args(parser):
        parser.add_argument("--frontend_module", type=str, default="espnet2.enh.separator.tfgridnet_separator")
        parser.add_argument("--frontend_class", type=str, default="TFGridNet")
        return parser

    def __init__(
        self,
        frontend_root: Optional[str] = None,
        frontend_module: Optional[str] = None,
        frontend_class: Optional[str] = None,
        frontend_config: Optional[str] = None,
        frontend_ckpt: Optional[str] = None,
        frontend_state_dict_key: Optional[str] = None,
        **kwargs,
    ):
        super().__init__()

        if frontend_root:
            root = str(frontend_root)
            if root not in sys.path:
                sys.path.insert(0, root)

        module_name = frontend_module or "espnet2.enh.separator.tfgridnet_separator"
        class_name = frontend_class or "TFGridNet"
        cfg = _load_yaml_or_json(frontend_config)

        # enhancement 场景必须是 1 源；input_dim 在官方实现里是 placeholder
        cfg.setdefault("input_dim", 0)
        cfg.setdefault("n_srcs", 1)
        cfg.setdefault("n_imics", 1)

        module = importlib.import_module(module_name)
        cls = getattr(module, class_name)
        self.model = cls(**cfg)

        if frontend_ckpt:
            ckpt = torch.load(frontend_ckpt, map_location="cpu", weights_only=False)
            state = strip_state_dict_prefixes(_guess_state_dict(ckpt, frontend_state_dict_key))
            missing, unexpected = self.model.load_state_dict(state, strict=False)
            if missing or unexpected:
                print(
                    f"[{self.__class__.__name__}] load_state_dict strict=False: missing={len(missing)} unexpected={len(unexpected)}",
                    flush=True,
                )

    def forward_waveform(self, noisy_wav: torch.Tensor) -> torch.Tensor:
        noisy_wav = ensure_mono_waveform(noisy_wav)
        noisy_bt = noisy_wav.squeeze(1)  # [B, T]
        ilens = torch.full(
            (noisy_bt.size(0),),
            noisy_bt.size(1),
            dtype=torch.long,
            device=noisy_bt.device,
        )
        enhanced_list, _, _ = self.model(noisy_bt, ilens)
        if not isinstance(enhanced_list, (list, tuple)) or len(enhanced_list) == 0:
            raise RuntimeError("TFGridNet did not return a non-empty enhanced source list")
        wav = ensure_mono_waveform(enhanced_list[0])
        return match_length(wav, noisy_wav)
