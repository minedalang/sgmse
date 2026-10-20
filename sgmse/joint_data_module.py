"""sgmse.joint_data_module

端到端联合训练（SEMamba -> SGMSE）的 DataModule。

- Dataset 只返回 (clean_wav, noisy_wav, spk_id)
- STFT/频谱变换在模型里做，保证 SEMamba 输出到 SGMSE 的 y_cond 可微

支持两种数据源：
1) base_dir 目录：base_dir/train|valid|test/{clean,noisy}/*.wav
2) wav.scp：每行 "uttid path"。clean 通过 clean_scp 或 clean_dir+template 解析

裁剪长度仍与 SGMSE 一致：target_len = (num_frames - 1) * hop_length
"""

from __future__ import annotations

import os
from glob import glob
from os.path import join
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchaudio import load,save
from sgmse.sampler import SpeakerBatchSampler


def get_window(window_type: str, window_length: int) -> torch.Tensor:
    if window_type == "sqrthann":
        return torch.sqrt(torch.hann_window(window_length, periodic=True))
    if window_type == "hann":
        return torch.hann_window(window_length, periodic=True)
    raise NotImplementedError(f"Window type {window_type} not implemented!")


def read_kaldi_scp(scp_path: str) -> List[Tuple[str, str]]:
    """Read Kaldi/ESPnet wav.scp style: <uttid> <path>."""
    scp_path = str(scp_path)
    base = Path(scp_path).parent
    items: List[Tuple[str, str]] = []
    with open(scp_path, "r", encoding="utf-8") as f:
        for ln, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            parts = line.split(maxsplit=1)
            if len(parts) != 2:
                raise ValueError(f"Bad scp line {ln} in {scp_path}: {line}")
            utt, p = parts[0], parts[1]
            p = str((base / p).resolve()) if not os.path.isabs(p) else p
            items.append((utt, p))
    return items


def scp_to_dict(items: List[Tuple[str, str]]) -> Dict[str, str]:
    return {k: v for k, v in items}


def _load_mono(path: str, expected_sr: Optional[int] = None) -> torch.Tensor:
    wav, sr = load(path)  # [C, T]
    if expected_sr is not None and sr != expected_sr:
        raise ValueError(f"Sample rate mismatch for {path}: got {sr}, expected {expected_sr}")
    if wav.dim() == 2 and wav.size(0) > 1:
        wav = wav[:1, :]
    return wav



class WavePairs(Dataset):
    """base_dir 数据集：返回 (clean_wav, noisy_wav, spk_id).

    clean/noisy 会被裁剪到固定长度，并做同尺度归一化。
    """

    def __init__(
        self,
        base_dir: str,
        subset: str,
        dummy: bool,
        shuffle_spec: bool,
        num_frames: int,
        hop_length: int,
        format: str = "default",
        normalize: str = "noisy",
        expected_sr: Optional[int] = None,
        **ignored_kwargs,
    ):
        super().__init__()
        self.base_dir = base_dir
        self.subset = subset
        self.dummy = dummy
        self.shuffle = shuffle_spec
        self.num_frames = num_frames
        self.hop_length = hop_length
        self.normalize = normalize
        self.expected_sr = expected_sr

        if format == "default":
            self.clean_files = sorted(glob(join(base_dir, subset, "clean", "**", "*.wav"), recursive=True))
            self.noisy_files = sorted(glob(join(base_dir, subset, "noisy", "**", "*.wav"), recursive=True))
        elif format == "reverb":
            self.clean_files = sorted(glob(join(base_dir, subset, "anechoic", "**", "*.wav"), recursive=True))
            self.noisy_files = sorted(glob(join(base_dir, subset, "reverb", "**", "*.wav"), recursive=True))
        else:
            raise NotImplementedError(f"Directory format {format} unknown!")

        if len(self.clean_files) != len(self.noisy_files):
            raise RuntimeError(
                f"Mismatch clean/noisy counts: {len(self.clean_files)} vs {len(self.noisy_files)} "
                f"(base_dir={base_dir}, subset={subset})"
            )

        # Needed by SpeakerBatchSampler. Unknown speakers are assigned unique
        # negative ids so unrelated utterances are never treated as positives.
        self.spk_ids = []
        for idx, path in enumerate(self.clean_files):
            sid = self._parse_spk_id(path)
            self.spk_ids.append(sid if sid != 0 else -(idx + 1))

    def __len__(self):
        return max(1, int(len(self.clean_files) / 200)) if self.dummy else len(self.clean_files)

    def _parse_spk_id(self, filepath: str) -> int:
        # 兼容原实现：p06709_xxx.wav -> 6709
        name = Path(filepath).name
        if name.startswith("p") and "_" in name:
            head = name.split("_")[0][1:]
            if head.isdigit():
                return int(head)
        return 0

    def __getitem__(self, i: int):
        
        x = _load_mono(self.clean_files[i], expected_sr=self.expected_sr)
        y = _load_mono(self.noisy_files[i], expected_sr=self.expected_sr)
        spk_id = self.spk_ids[i]

        # 先对齐长度，避免 clean/noisy 长度略有差异
        min_len = min(x.size(-1), y.size(-1))
        x = x[..., :min_len]
        y = y[..., :min_len]

        # 裁剪部分用于训练
        target_len = (self.num_frames - 1) * self.hop_length

        cur_len = x.size(-1)  # 这里已经做过 min_len 对齐了
        pad = max(target_len - cur_len, 0)

        if pad == 0:
            # 同一个 start 同时裁 clean/noisy（关键修复点）
            if self.shuffle and cur_len > target_len:
                start = int(np.random.uniform(0, cur_len - target_len))
            else:
                start = int((cur_len - target_len) / 2)
            x = x[..., start : start + target_len]
            y = y[..., start : start + target_len]
        else:
            # 同样的 pad 同时 pad clean/noisy
            left = pad // 2
            right = left + (pad % 2)
            x = F.pad(x, (left, right), mode="constant", value=0.0)
            y = F.pad(y, (left, right), mode="constant", value=0.0)

        # 归一化，保证 x/y 在同一尺度
        if self.normalize == "noisy":
            normfac = y.abs().max()
        elif self.normalize == "clean":
            normfac = x.abs().max()
        else:
            normfac = torch.tensor(1.0, dtype=x.dtype, device=x.device)
        normfac = normfac.clamp_min(1e-12)
        x = x / normfac
        y = y / normfac

        return x, y, torch.tensor(spk_id, dtype=torch.long)


class WavePairsScp(Dataset):
    """wav.scp 数据集：返回 (clean_wav, noisy_wav, spk_id)."""

    def __init__(
        self,
        wav_scp: str,
        clean_dir: str,
        subset: str,
        dummy: bool,
        shuffle_spec: bool,
        num_frames: int,
        hop_length: int,
        normalize: str = "noisy",
        clean_template: str = "clean_fileid_{fileid}.wav",
        clean_scp: Optional[str] = None,
        expected_sr: Optional[int] = None,
        **ignored_kwargs,
    ):
        super().__init__()
        import re

        self._re_fileid = re.compile(r"fileid_(\d+)")
        self._re_fileid_any = re.compile(r"(\d{3,})")
        self._re_reader = re.compile(r"reader_(\d+)")

        self.clean_dir = Path(clean_dir)
        self.clean_template = clean_template

        self.wav_items = read_kaldi_scp(wav_scp)
        self.clean_map = scp_to_dict(read_kaldi_scp(clean_scp)) if clean_scp else None

        self.uttids = [u for u, _ in self.wav_items]
        self.noisy_files = [p for _, p in self.wav_items]
        self.dummy = dummy
        self.shuffle = shuffle_spec
        self.num_frames = num_frames
        self.hop_length = hop_length
        self.normalize = normalize
        self.expected_sr = expected_sr
        self.subset = subset

        # Precompute ids for speaker-balanced contrastive batches. If parsing
        # fails, keep that utterance unique rather than collapsing every unknown
        # speaker to id 0 (which would create false positives).
        self.spk_ids = []
        for idx, (utt, noisy_path) in enumerate(self.wav_items):
            sid = self._parse_spk_id(utt, noisy_path)
            self.spk_ids.append(sid if sid != 0 else -(idx + 1))

    def __len__(self):
        return max(1, int(len(self.wav_items) / 200)) if self.dummy else len(self.wav_items)

    def _parse_fileid(self, utt: str, noisy_path: str) -> int:
        base = Path(noisy_path).name
        m = self._re_fileid.search(base)
        if m:
            return int(m.group(1))
        m2 = self._re_fileid_any.search(utt)
        if m2:
            return int(m2.group(1))
        raise ValueError(f"Cannot parse fileid from utt={utt}, noisy={noisy_path}")

    def _parse_spk_id(self, utt: str, noisy_path: str) -> int:
        m = self._re_reader.search(utt)
        if m:
            return int(m.group(1))
        m = self._re_reader.search(Path(noisy_path).name)
        if m:
            return int(m.group(1))
        return 0

    def _resolve_clean_path(self, utt: str, noisy_path: str) -> str:
        if self.clean_map is not None and utt in self.clean_map:
            return self.clean_map[utt]
        fileid = self._parse_fileid(utt, noisy_path)
        return str(self.clean_dir / self.clean_template.format(fileid=fileid))

    def __getitem__(self, i: int):

        utt = self.uttids[i]
        noisy_path = self.noisy_files[i]
        clean_path = self._resolve_clean_path(utt, noisy_path)
        spk_id = self.spk_ids[i]

        y = _load_mono(noisy_path, expected_sr=self.expected_sr)
        x = _load_mono(clean_path, expected_sr=self.expected_sr)

        min_len = min(x.size(-1), y.size(-1))
        x = x[..., :min_len]
        y = y[..., :min_len]

        target_len = (self.num_frames - 1) * self.hop_length
        cur_len = x.size(-1)  # 已经 min_len 对齐
        pad = max(target_len - cur_len, 0)

        if pad == 0:
            if self.shuffle and cur_len > target_len:
                start = int(np.random.uniform(0, cur_len - target_len))
            else:
                start = int((cur_len - target_len) / 2)
            x = x[..., start : start + target_len]
            y = y[..., start : start + target_len]
        else:
            left = pad // 2
            right = left + (pad % 2)
            x = F.pad(x, (left, right), mode="constant", value=0.0)
            y = F.pad(y, (left, right), mode="constant", value=0.0)

        if self.normalize == "noisy":
            normfac = y.abs().max()
        elif self.normalize == "clean":
            normfac = x.abs().max()
        else:
            normfac = torch.tensor(1.0, dtype=x.dtype, device=x.device)
        normfac = normfac.clamp_min(1e-12)
        x = x / normfac
        y = y / normfac

        return x, y, torch.tensor(spk_id, dtype=torch.long)


class _BaseJointWaveDataModule(pl.LightningDataModule):
    """提供 stft/spec_transform 工具方法，与原 SpecsDataModule 保持一致。"""

    def __init__(
        self,
        batch_size: int = 8,
        n_fft: int = 510,
        hop_length: int = 128,
        num_frames: int = 256,
        window: str = "hann",
        num_workers: int = 4,
        dummy: bool = False,
        spec_factor: float = 0.15,
        spec_abs_exponent: float = 0.5,
        normalize: str = "noisy",
        transform_type: str = "exponent",
        gpu: bool = True,
        **kwargs,
    ):
        super().__init__()
        self.batch_size = batch_size
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.num_frames = num_frames
        self.window = get_window(window, self.n_fft)
        self.num_workers = num_workers
        self.dummy = dummy
        self.spec_factor = spec_factor
        self.spec_abs_exponent = spec_abs_exponent
        self.normalize = normalize
        self.transform_type = transform_type
        self.gpu = gpu
        self.kwargs = kwargs

    @property
    def istft_kwargs(self) -> dict:
        return dict(n_fft=self.n_fft, hop_length=self.hop_length, window=self.window, center=True)

    @property
    def stft_kwargs(self) -> dict:
        return {**self.istft_kwargs, "return_complex": True}

    def spec_fwd(self, spec: torch.Tensor) -> torch.Tensor:
        if self.transform_type == "exponent":
            if self.spec_abs_exponent != 1:
                e = self.spec_abs_exponent
                # spec = spec.abs() ** e * torch.exp(1j * spec.angle())
                # 在联合训练中，反响传播时spec_fwd可能会产生NaN，需要改为下面这种方式
                eps = 1e-8  # 关键：防止 mag=0 导致梯度爆炸，可在 1e-6~1e-3 之间调
                mag = torch.sqrt(spec.real * spec.real + spec.imag * spec.imag + eps)
                phase = spec / mag.clamp(min=1e-8)
                spec = mag.pow(e) * phase
                if (not torch.isfinite(spec.real).all()) or (not torch.isfinite(spec.imag).all()):
                    real = torch.nan_to_num(spec.real, nan=0.0, posinf=0.0, neginf=0.0)
                    imag = torch.nan_to_num(spec.imag, nan=0.0, posinf=0.0, neginf=0.0)
                    spec = torch.complex(real, imag)

            spec = spec * self.spec_factor
        elif self.transform_type == "log":
            spec = torch.log(1 + spec.abs()) * torch.exp(1j * spec.angle())
            spec = spec * self.spec_factor
        elif self.transform_type == "none":
            pass
        return spec

    def spec_back(self, spec: torch.Tensor) -> torch.Tensor:
        if self.transform_type == "exponent":
            spec = spec / self.spec_factor
            if self.spec_abs_exponent != 1:
                e = self.spec_abs_exponent
                spec = spec.abs() ** (1 / e) * torch.exp(1j * spec.angle())
        elif self.transform_type == "log":
            spec = spec / self.spec_factor
            spec = (torch.exp(spec.abs()) - 1) * torch.exp(1j * spec.angle())
        elif self.transform_type == "none":
            pass
        return spec
    def stft(self, sig: torch.Tensor) -> torch.Tensor:
        """
        输入:
        sig: [T] 或 [B,T] 或 [B,1,T]
        输出:
        spec: complex, 形状 [F,T] 或 [B,F,T]
        """
        if sig.dim() == 3 and sig.size(1) == 1:
            sig = sig[:, 0, :]
        if sig.dim() not in (1, 2):
            raise RuntimeError(f"stft expects [T] or [B,T] or [B,1,T], got {tuple(sig.shape)}")

        window = self.window.to(sig.device)
        return torch.stft(
            sig,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.n_fft,
            window=window,
            center=True,
            return_complex=True,
        )

    def istft(self, spec: torch.Tensor, length: int | None = None) -> torch.Tensor:
        """
        输入:
        spec: complex, [F,T] 或 [B,F,T] 或 [B,1,F,T]
        输出:
        wav: [T] 或 [B,T]
        """
        if spec.dim() == 4 and spec.size(1) == 1:
            spec = spec[:, 0, :, :]
        if spec.dim() not in (2, 3):
            raise RuntimeError(f"istft expects [F,T] or [B,F,T] or [B,1,F,T], got {tuple(spec.shape)}")

        window = self.window.to(spec.device)
        return torch.istft(
            spec,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.n_fft,
            window=window,
            center=True,
            length=length,
        )

    def train_dataloader(
        self,
        contrastive_weight: float = 0.0,
        *args,
        speaker_balanced_batches: bool = False,
        **kwargs,
    ):
        """Build the training loader.

        Speaker-balanced batches are used for contrastive loss, or when
        explicitly requested for a matched-speaker sampling ablation. Every
        selected speaker contributes at least two utterances.
        """
        if contrastive_weight > 0.0 or speaker_balanced_batches:
            if not hasattr(self.train_set, "spk_ids"):
                raise RuntimeError(
                    "Speaker-balanced training requires train_set.spk_ids."
                )
            batch_sampler = SpeakerBatchSampler(
                spk_ids=self.train_set.spk_ids,
                batch_size=self.batch_size,
                min_per_speaker=2,
                drop_last=True,
            )
            if len(batch_sampler) == 0:
                raise RuntimeError(
                    "No speaker-balanced batch can be formed. Check speaker-id parsing "
                    "or use a dataset with at least two utterances per speaker."
                )
            return DataLoader(
                self.train_set,
                batch_sampler=batch_sampler,
                num_workers=self.num_workers,
                pin_memory=self.gpu,
            )

        return DataLoader(
            self.train_set,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.gpu,
            shuffle=True,
            drop_last=True,
        )


    def val_dataloader(self):
        return DataLoader(
            self.valid_set,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.gpu,
            shuffle=False,
            drop_last=False,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_set,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.gpu,
            shuffle=False,
            drop_last=False,
        )


class JointWaveDataModule(_BaseJointWaveDataModule):
    """base_dir 版本。"""

    @staticmethod
    def add_argparse_args(parser):
        parser.add_argument("--base_dir", type=str, required=True)
        parser.add_argument("--format", type=str, choices=("default", "reverb"), default="default")
        parser.add_argument("--batch_size", type=int, default=8)
        parser.add_argument("--n_fft", type=int, default=510)
        parser.add_argument("--hop_length", type=int, default=128)
        parser.add_argument("--num_frames", type=int, default=256)
        parser.add_argument("--window", type=str, choices=("sqrthann", "hann"), default="hann")
        parser.add_argument("--num_workers", type=int, default=4)
        parser.add_argument("--dummy", action="store_true")
        parser.add_argument("--spec_factor", type=float, default=0.15)
        parser.add_argument("--spec_abs_exponent", type=float, default=0.5)
        parser.add_argument("--normalize", type=str, choices=("clean", "noisy", "not"), default="noisy")
        parser.add_argument("--transform_type", type=str, choices=("exponent", "log", "none"), default="exponent")
        parser.add_argument("--expected_sr", type=int, default=None)
        return parser

    def __init__(self, base_dir: str, format: str = "default", expected_sr: Optional[int] = None, **kwargs):
        super().__init__(**kwargs)
        self.base_dir = base_dir
        self.format = format
        self.expected_sr = expected_sr

    def setup(self, stage=None):
        ds_kwargs = dict(
            dummy=self.dummy,
            shuffle_spec=True,
            num_frames=self.num_frames,
            hop_length=self.hop_length,
            format=self.format,
            normalize=self.normalize,
            expected_sr=self.expected_sr,
        )
        if stage in (None, "fit"):
            self.train_set = WavePairs(self.base_dir, "train", **ds_kwargs)
            ds_kwargs["shuffle_spec"] = False
            self.valid_set = WavePairs(self.base_dir, "valid", **ds_kwargs)
        if stage in (None, "test"):
            ds_kwargs["shuffle_spec"] = False
            self.test_set = WavePairs(self.base_dir, "test", **ds_kwargs)


class JointWaveScpDataModule(_BaseJointWaveDataModule):
    """wav.scp 版本。"""

    @staticmethod
    def add_argparse_args(parser):
        parser.add_argument("--train_wav_scp", type=str, required=True)
        parser.add_argument("--valid_wav_scp", type=str, required=True)
        parser.add_argument("--test_wav_scp", type=str, default=None)
        parser.add_argument("--train_clean_scp", type=str, default=None)
        parser.add_argument("--valid_clean_scp", type=str, default=None)
        parser.add_argument("--test_clean_scp", type=str, default=None)
        parser.add_argument("--clean_dir", type=str, required=True)
        parser.add_argument("--clean_template", type=str, default="clean_fileid_{fileid}.wav")
        parser.add_argument("--batch_size", type=int, default=8)
        parser.add_argument("--n_fft", type=int, default=510)
        parser.add_argument("--hop_length", type=int, default=128)
        parser.add_argument("--num_frames", type=int, default=256)
        parser.add_argument("--window", type=str, choices=("sqrthann", "hann"), default="hann")
        parser.add_argument("--num_workers", type=int, default=4)
        parser.add_argument("--dummy", action="store_true")
        parser.add_argument("--spec_factor", type=float, default=0.15)
        parser.add_argument("--spec_abs_exponent", type=float, default=0.5)
        parser.add_argument("--normalize", type=str, choices=("clean", "noisy", "not"), default="noisy")
        parser.add_argument("--transform_type", type=str, choices=("exponent", "log", "none"), default="exponent")
        parser.add_argument("--expected_sr", type=int, default=None)
        return parser

    def __init__(
        self,
        train_wav_scp: str,
        valid_wav_scp: str,
        clean_dir: str,
        clean_template: str = "clean_fileid_{fileid}.wav",
        test_wav_scp: Optional[str] = None,
        train_clean_scp: Optional[str] = None,
        valid_clean_scp: Optional[str] = None,
        test_clean_scp: Optional[str] = None,
        expected_sr: Optional[int] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.train_wav_scp = train_wav_scp
        self.valid_wav_scp = valid_wav_scp
        self.test_wav_scp = test_wav_scp
        self.train_clean_scp = train_clean_scp
        self.valid_clean_scp = valid_clean_scp
        self.test_clean_scp = test_clean_scp
        self.clean_dir = clean_dir
        self.clean_template = clean_template
        self.expected_sr = expected_sr

    def setup(self, stage=None):
        ds_kwargs = dict(
            clean_dir=self.clean_dir,
            subset="train",
            dummy=self.dummy,
            shuffle_spec=True,
            num_frames=self.num_frames,
            hop_length=self.hop_length,
            normalize=self.normalize,
            clean_template=self.clean_template,
            expected_sr=self.expected_sr,
        )
        if stage in (None, "fit"):
            self.train_set = WavePairsScp(
                wav_scp=self.train_wav_scp,
                clean_scp=self.train_clean_scp,
                **ds_kwargs,
            )
            ds_kwargs["subset"] = "valid"
            ds_kwargs["shuffle_spec"] = False
            self.valid_set = WavePairsScp(
                wav_scp=self.valid_wav_scp,
                clean_scp=self.valid_clean_scp,
                **ds_kwargs,
            )
        if stage in (None, "test"):
            if self.test_wav_scp is None:
                # 允许不提供 test
                self.test_set = self.valid_set
            else:
                ds_kwargs["subset"] = "test"
                ds_kwargs["shuffle_spec"] = False
                self.test_set = WavePairsScp(
                    wav_scp=self.test_wav_scp,
                    clean_scp=self.test_clean_scp,
                    **ds_kwargs,
                )


class JointWaveDataModule(_BaseJointWaveDataModule):
    """base_dir 版本."""

    @staticmethod
    def add_argparse_args(parser):
        parser.add_argument("--base_dir", type=str, required=True,
                            help="Dataset base dir with train/valid/test subfolders")
        parser.add_argument("--format", type=str, choices=("default", "reverb"), default="default")
        parser.add_argument("--batch_size", type=int, default=8)
        parser.add_argument("--n_fft", type=int, default=510)
        parser.add_argument("--hop_length", type=int, default=128)
        parser.add_argument("--num_frames", type=int, default=256)
        parser.add_argument("--window", type=str, choices=("sqrthann", "hann"), default="hann")
        parser.add_argument("--num_workers", type=int, default=4)
        parser.add_argument("--dummy", action="store_true")
        parser.add_argument("--spec_factor", type=float, default=0.15)
        parser.add_argument("--spec_abs_exponent", type=float, default=0.5)
        parser.add_argument("--normalize", type=str, choices=("clean", "noisy", "not"), default="noisy")
        parser.add_argument("--transform_type", type=str, choices=("exponent", "log", "none"), default="exponent")
        parser.add_argument("--expected_sr", type=int, default=None)
        return parser

    def __init__(self, base_dir: str, format: str = "default", expected_sr: Optional[int] = None, **kwargs):
        super().__init__(**kwargs)
        self.base_dir = base_dir
        self.format = format
        self.expected_sr = expected_sr

    def setup(self, stage=None):
        ds_kwargs = dict(
            dummy=self.dummy,
            num_frames=self.num_frames,
            hop_length=self.hop_length,
            format=self.format,
            normalize=self.normalize,
            expected_sr=self.expected_sr,
        )
        if stage in (None, "fit"):
            self.train_set = WavePairs(self.base_dir, subset="train", shuffle_spec=True, **ds_kwargs)
            self.valid_set = WavePairs(self.base_dir, subset="valid", shuffle_spec=False, **ds_kwargs)
        if stage in (None, "test"):
            self.test_set = WavePairs(self.base_dir, subset="test", shuffle_spec=False, **ds_kwargs)


class JointWaveScpDataModule(_BaseJointWaveDataModule):
    """wav.scp 版本."""

    @staticmethod
    def add_argparse_args(parser):
        parser.add_argument("--train_wav_scp", type=str, required=True)
        parser.add_argument("--valid_wav_scp", type=str, required=True)
        parser.add_argument("--test_wav_scp", type=str, default=None)
        parser.add_argument("--train_clean_scp", type=str, default=None)
        parser.add_argument("--valid_clean_scp", type=str, default=None)
        parser.add_argument("--test_clean_scp", type=str, default=None)
        parser.add_argument("--clean_dir", type=str, required=True)
        parser.add_argument("--clean_template", type=str, default="clean_fileid_{fileid}.wav")
        parser.add_argument("--batch_size", type=int, default=8)
        parser.add_argument("--n_fft", type=int, default=510)
        parser.add_argument("--hop_length", type=int, default=128)
        parser.add_argument("--num_frames", type=int, default=256)
        parser.add_argument("--window", type=str, choices=("sqrthann", "hann"), default="hann")
        parser.add_argument("--num_workers", type=int, default=4)
        parser.add_argument("--dummy", action="store_true")
        parser.add_argument("--spec_factor", type=float, default=0.15)
        parser.add_argument("--spec_abs_exponent", type=float, default=0.5)
        parser.add_argument("--normalize", type=str, choices=("clean", "noisy", "not"), default="noisy")
        parser.add_argument("--transform_type", type=str, choices=("exponent", "log", "none"), default="exponent")
        parser.add_argument("--expected_sr", type=int, default=None)
        return parser

    def __init__(
        self,
        train_wav_scp: str,
        valid_wav_scp: str,
        clean_dir: str,
        clean_template: str = "clean_fileid_{fileid}.wav",
        test_wav_scp: Optional[str] = None,
        train_clean_scp: Optional[str] = None,
        valid_clean_scp: Optional[str] = None,
        test_clean_scp: Optional[str] = None,
        expected_sr: Optional[int] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.train_wav_scp = train_wav_scp
        self.valid_wav_scp = valid_wav_scp
        self.test_wav_scp = test_wav_scp
        self.clean_dir = clean_dir
        self.clean_template = clean_template
        self.train_clean_scp = train_clean_scp
        self.valid_clean_scp = valid_clean_scp
        self.test_clean_scp = test_clean_scp
        self.expected_sr = expected_sr

    def setup(self, stage=None):
        ds_common = dict(
            dummy=self.dummy,
            num_frames=self.num_frames,
            hop_length=self.hop_length,
            normalize=self.normalize,
            clean_dir=self.clean_dir,
            clean_template=self.clean_template,
            expected_sr=self.expected_sr,
        )
        if stage in (None, "fit"):
            self.train_set = WavePairsScp(
                wav_scp=self.train_wav_scp,
                clean_scp=self.train_clean_scp,
                subset="train",
                shuffle_spec=True,
                **ds_common,
            )
            self.valid_set = WavePairsScp(
                wav_scp=self.valid_wav_scp,
                clean_scp=self.valid_clean_scp,
                subset="valid",
                shuffle_spec=False,
                **ds_common,
            )
        if stage in (None, "test"):
            if self.test_wav_scp is None:
                self.test_set = self.valid_set
            else:
                self.test_set = WavePairsScp(
                    wav_scp=self.test_wav_scp,
                    clean_scp=self.test_clean_scp,
                    subset="test",
                    shuffle_spec=False,
                    **ds_common,
                )
