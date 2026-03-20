from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path
from typing import Any, Dict, Optional

import torch

from .base import BaseFrontend, ensure_mono_waveform, match_length, strip_state_dict_prefixes, try_get_tensor_from_output


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


def _import_from_root(root: Optional[str], module_name: str):
    if root:
        root = str(root)
        if root not in sys.path:
            sys.path.insert(0, root)
    return importlib.import_module(module_name)


def _guess_state_dict(ckpt: Any, state_dict_key: Optional[str] = None) -> Dict[str, Any]:
    if isinstance(ckpt, dict):
        if state_dict_key and state_dict_key in ckpt:
            return ckpt[state_dict_key]
        for key in ("state_dict", "model", "generator", "net"):
            if key in ckpt and isinstance(ckpt[key], dict):
                return ckpt[key]
    if isinstance(ckpt, dict):
        return ckpt
    raise RuntimeError("Cannot parse checkpoint into a state_dict")


class ExternalWaveformFrontend(BaseFrontend):
    """通用外部时域前端包装器。

    适用于 Storm / TF-GridNet 这类“输入波形 -> 输出波形”的判别式前端。
    约定：
    - 能从 `frontend_root + frontend_module` import 到类或构建函数。
    - forward 接受 `[B,T]` 或 `[B,1,T]`。
    - 返回 tensor / tuple / dict 中包含增强波形。
    """

    default_module_name: str = ""
    default_class_name: str = ""

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
        module_name = frontend_module or self.default_module_name
        class_name = frontend_class or self.default_class_name
        if not module_name or not class_name:
            raise ValueError("frontend_module and frontend_class are required for external waveform frontends")

        cfg = _load_yaml_or_json(frontend_config)
        module = _import_from_root(frontend_root, module_name)
        builder = getattr(module, class_name)

        try:
            self.model = builder(cfg)
        except TypeError:
            try:
                self.model = builder(**cfg)
            except TypeError:
                if cfg:
                    raise
                self.model = builder()

        if frontend_ckpt:
            ckpt = torch.load(frontend_ckpt, map_location="cpu", weights_only=False)
            state = _guess_state_dict(ckpt, frontend_state_dict_key)
            state = strip_state_dict_prefixes(state)
            missing, unexpected = self.model.load_state_dict(state, strict=False)
            if missing or unexpected:
                print(
                    f"[{self.__class__.__name__}] load_state_dict strict=False: "
                    f"missing={len(missing)} unexpected={len(unexpected)}",
                    flush=True,
                )

    def forward_waveform(self, noisy_wav: torch.Tensor) -> torch.Tensor:
        noisy_wav = ensure_mono_waveform(noisy_wav)
        try:
            out = self.model(noisy_wav)
        except TypeError:
            out = self.model(noisy_wav.squeeze(1))
        wav = try_get_tensor_from_output(out)
        wav = ensure_mono_waveform(wav)
        return match_length(wav, noisy_wav)
