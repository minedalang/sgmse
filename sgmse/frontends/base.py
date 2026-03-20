from __future__ import annotations

from typing import Any, Dict

import torch
import torch.nn as nn


class BaseFrontend(nn.Module):
    """统一的前端接口。

    所有前端都接收波形 `[B, 1, T]`，返回波形 `[B, 1, T']`。
    joint/pretrain 侧只依赖这个接口，不关心前端内部是 STFT 模型还是时域模型。
    """

    supports_direct_waveform: bool = True

    @staticmethod
    def add_argparse_args(parser):
        return parser

    def forward_waveform(self, noisy_wav: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def forward(self, noisy_wav: torch.Tensor) -> torch.Tensor:
        return self.forward_waveform(noisy_wav)

    def freeze(self):
        for p in self.parameters():
            p.requires_grad = False
        self.eval()
        return self

    def unfreeze(self):
        for p in self.parameters():
            p.requires_grad = True
        self.train()
        return self


def ensure_mono_waveform(x: torch.Tensor) -> torch.Tensor:
    if x.dim() == 2:
        x = x.unsqueeze(1)
    if x.dim() != 3:
        raise RuntimeError(f"Expected waveform [B,1,T] or [B,T], got {tuple(x.shape)}")
    if x.size(1) != 1:
        x = x[:, :1, :]
    return x


def match_length(y: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
    """把 y 的最后一维长度调成 ref 的长度（center crop/pad）。"""
    y = ensure_mono_waveform(y)
    ref = ensure_mono_waveform(ref)
    target_len = ref.size(-1)
    cur_len = y.size(-1)
    if cur_len == target_len:
        return y
    if cur_len > target_len:
        start = (cur_len - target_len) // 2
        return y[..., start:start + target_len]
    pad = target_len - cur_len
    left = pad // 2
    right = pad - left
    return torch.nn.functional.pad(y, (left, right), mode="constant", value=0.0)


def try_get_tensor_from_output(output: Any) -> torch.Tensor:
    """兼容外部模型的常见返回格式。"""
    if torch.is_tensor(output):
        return output
    if isinstance(output, (tuple, list)) and len(output) > 0:
        for item in output:
            if torch.is_tensor(item):
                return item
    if isinstance(output, dict):
        for key in ("wav", "waveform", "audio", "enhanced", "output", "pred"):
            if key in output and torch.is_tensor(output[key]):
                return output[key]
    raise RuntimeError(f"Cannot extract waveform tensor from frontend output of type {type(output)!r}")


def strip_state_dict_prefixes(state_dict: Dict[str, Any]) -> Dict[str, Any]:
    out = {}
    for k, v in state_dict.items():
        new_key = k
        for prefix in ("module.", "model.", "generator.", "net."):
            if new_key.startswith(prefix):
                new_key = new_key[len(prefix):]
        out[new_key] = v
    return out
