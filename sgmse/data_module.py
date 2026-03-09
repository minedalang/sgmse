
from os.path import join,exists
import os
from pathlib import Path
import torch
import pytorch_lightning as pl
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from glob import glob
from torchaudio import load
import numpy as np
import torch.nn.functional as F
import logging
import random
from collections import defaultdict
from sgmse.sampler import SpeakerBatchSampler 
import re
from typing import Callable, Dict, List, Optional, Tuple

def get_window(window_type, window_length):
    if window_type == 'sqrthann':
        return torch.sqrt(torch.hann_window(window_length, periodic=True))
    elif window_type == 'hann':
        return torch.hann_window(window_length, periodic=True)
    else:
        raise NotImplementedError(f"Window type {window_type} not implemented!")
def read_kaldi_scp(scp_path: str) -> List[Tuple[str, str]]:
    """
    Read Kaldi/ESPnet wav.scp / spk1.scp style:
      <uttid> <path>
    Returns list[(uttid, path)] preserving file order.
    """
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

class Specs(Dataset):
    def __init__(self, data_dir, subset, dummy, shuffle_spec, num_frames,
            format='default', normalize="noisy", spec_transform=None,
            stft_kwargs=None, **ignored_kwargs):

        
        # 设置日志记录器
        logging.basicConfig(
            filename='training_log.txt',  # 日志文件
            level=logging.INFO,           # 日志级别
            format='%(asctime)s - %(message)s'  # 日志格式
        )

        # Read file paths according to file naming format.
        if format == "default":
            self.clean_files = []
            self.clean_files += sorted(glob(join(data_dir, subset, "clean", "*.wav")))
            self.clean_files += sorted(glob(join(data_dir, subset, "clean", "**", "*.wav")))
            self.noisy_files = []
            self.noisy_files += sorted(glob(join(data_dir, subset, "noisy", "*.wav")))
            self.noisy_files += sorted(glob(join(data_dir, subset, "noisy", "**", "*.wav")))
        elif format == "reverb":
            self.clean_files = []
            self.clean_files += sorted(glob(join(data_dir, subset, "anechoic", "*.wav")))
            self.clean_files += sorted(glob(join(data_dir, subset, "anechoic", "**", "*.wav")))
            self.noisy_files = []
            self.noisy_files += sorted(glob(join(data_dir, subset, "reverb", "*.wav")))
            self.noisy_files += sorted(glob(join(data_dir, subset, "reverb", "**", "*.wav")))
        else:
            # Feel free to add your own directory format
            raise NotImplementedError(f"Directory format {format} unknown!")

        # 创建说话人ID到样本的映射
        self.spk_to_samples = defaultdict(list)
        for clean_file, noisy_file in zip(self.clean_files, self.noisy_files):
            spk_id = self._get_speaker_id(clean_file)
            # 将样本根据说话人ID分类
            self.spk_to_samples[spk_id].append((clean_file, noisy_file, spk_id))  
        # 每个样本的说话人 id 列表
        self.spk_ids = [self._get_speaker_id(f) for f in self.clean_files]
        self.dummy = dummy
        self.num_frames = num_frames
        self.shuffle_spec = shuffle_spec
        self.normalize = normalize
        self.spec_transform = spec_transform

        assert all(k in stft_kwargs.keys() for k in ["n_fft", "hop_length", "center", "window"]), "misconfigured STFT kwargs"
        self.stft_kwargs = stft_kwargs
        self.hop_length = self.stft_kwargs["hop_length"]
        assert self.stft_kwargs.get("center", None) == True, "'center' must be True for current implementation"

    def __getitem__(self, i):
        x, _ = load(self.clean_files[i])
        y, _ = load(self.noisy_files[i])

        # 获取说话人信息
        spk_id = self._get_speaker_id(self.clean_files[i])

        # formula applies for center=True
        target_len = (self.num_frames - 1) * self.hop_length
        current_len = x.size(-1)
        pad = max(target_len - current_len, 0)
        if pad == 0:
            # extract random part of the audio file
            if self.shuffle_spec:
                start = int(np.random.uniform(0, current_len-target_len))
            else:
                start = int((current_len-target_len)/2)
            x = x[..., start:start+target_len]
            y = y[..., start:start+target_len]
        else:
            # pad audio if the length T is smaller than num_frames
            x = F.pad(x, (pad//2, pad//2+(pad%2)), mode='constant')
            y = F.pad(y, (pad//2, pad//2+(pad%2)), mode='constant')

        # normalize w.r.t to the noisy or the clean signal or not at all
        # to ensure same clean signal power in x and y.
        if self.normalize == "noisy":
            normfac = y.abs().max()
        elif self.normalize == "clean":
            normfac = x.abs().max()
        elif self.normalize == "not":
            normfac = 1.0
        x = x / normfac
        y = y / normfac

        X = torch.stft(x, **self.stft_kwargs)
        Y = torch.stft(y, **self.stft_kwargs)

        X, Y = self.spec_transform(X), self.spec_transform(Y)
        return X, Y, spk_id

    def _get_speaker_id(self, filepath):
        """
        通过文件路径获取说话人 ID，假设文件名或路径包含说话人信息。

        """
        # 示例：假设文件名格式为 p<spk_id>_<audio_id>.wav
        spk_id = filepath.split('/')[-1].split('_')[0][1:]  # 去掉 'p' 并提取数字部分
        return int(spk_id)

    def __len__(self):
        if self.dummy:
            # for debugging shrink the data set size
            return int(len(self.clean_files)/200)
        else:
            return len(self.clean_files)


class SpecsDataModule(pl.LightningDataModule):
    @staticmethod
    def add_argparse_args(parser):
        parser.add_argument("--base_dir", type=str, required=True, help="The base directory of the dataset. Should contain `train`, `valid` and `test` subdirectories, each of which contain `clean` and `noisy` subdirectories.")
        parser.add_argument("--format", type=str, choices=("default", "reverb"), default="default", help="Read file paths according to file naming format.")
        parser.add_argument("--batch_size", type=int, default=8, help="The batch size. 8 by default.")
        parser.add_argument("--n_fft", type=int, default=510, help="Number of FFT bins. 510 by default.")   # to assure 256 freq bins
        parser.add_argument("--hop_length", type=int, default=128, help="Window hop length. 128 by default.")
        parser.add_argument("--num_frames", type=int, default=256, help="Number of frames for the dataset. 256 by default.")
        parser.add_argument("--window", type=str, choices=("sqrthann", "hann"), default="hann", help="The window function to use for the STFT. 'hann' by default.")
        parser.add_argument("--num_workers", type=int, default=4, help="Number of workers to use for DataLoaders. 4 by default.")
        parser.add_argument("--dummy", action="store_true", help="Use reduced dummy dataset for prototyping.")
        parser.add_argument("--spec_factor", type=float, default=0.15, help="Factor to multiply complex STFT coefficients by. 0.15 by default.")
        parser.add_argument("--spec_abs_exponent", type=float, default=0.5, help="Exponent e for the transformation abs(z)**e * exp(1j*angle(z)). 0.5 by default.")
        parser.add_argument("--normalize", type=str, choices=("clean", "noisy", "not"), default="noisy", help="Normalize the input waveforms by the clean signal, the noisy signal, or not at all.")
        parser.add_argument("--transform_type", type=str, choices=("exponent", "log", "none"), default="exponent", help="Spectogram transformation for input representation.")
        return parser

    def __init__(
        self, base_dir, format='default', batch_size=8,
        n_fft=510, hop_length=128, num_frames=256, window='hann',
        num_workers=4, dummy=False, spec_factor=0.15, spec_abs_exponent=0.5,
        gpu=True, normalize='noisy', transform_type="exponent", **kwargs
    ):
        super().__init__()
        self.base_dir = base_dir
        self.format = format
        self.batch_size = batch_size
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.num_frames = num_frames
        self.window = get_window(window, self.n_fft)
        self.windows = {}
        self.num_workers = num_workers
        self.dummy = dummy
        self.spec_factor = spec_factor
        self.spec_abs_exponent = spec_abs_exponent
        self.gpu = gpu
        self.normalize = normalize
        self.transform_type = transform_type
        self.kwargs = kwargs

    def setup(self, stage=None):
        specs_kwargs = dict(
            stft_kwargs=self.stft_kwargs, num_frames=self.num_frames,
            spec_transform=self.spec_fwd, **self.kwargs
        )
        if stage == 'fit' or stage is None:
            self.train_set = Specs(data_dir=self.base_dir, subset='train',
                dummy=self.dummy, shuffle_spec=True, format=self.format,
                normalize=self.normalize, **specs_kwargs)
            self.valid_set = Specs(data_dir=self.base_dir, subset='valid',
                dummy=self.dummy, shuffle_spec=False, format=self.format,
                normalize=self.normalize, **specs_kwargs)
        if stage == 'test' or stage is None:
            self.test_set = Specs(data_dir=self.base_dir, subset='test',
                dummy=self.dummy, shuffle_spec=False, format=self.format,
                normalize=self.normalize, **specs_kwargs)

    def spec_fwd(self, spec):
        if self.transform_type == "exponent":
            if self.spec_abs_exponent != 1:
                # only do this calculation if spec_exponent != 1, otherwise it's quite a bit of wasted computation
                # and introduced numerical error
                e = self.spec_abs_exponent
                spec = spec.abs()**e * torch.exp(1j * spec.angle())
            spec = spec * self.spec_factor
        elif self.transform_type == "log":
            spec = torch.log(1 + spec.abs()) * torch.exp(1j * spec.angle())
            spec = spec * self.spec_factor
        elif self.transform_type == "none":
            spec = spec
        return spec

    def spec_back(self, spec):
        if self.transform_type == "exponent":
            spec = spec / self.spec_factor
            if self.spec_abs_exponent != 1:
                e = self.spec_abs_exponent
                spec = spec.abs()**(1/e) * torch.exp(1j * spec.angle())
        elif self.transform_type == "log":
            spec = spec / self.spec_factor
            spec = (torch.exp(spec.abs()) - 1) * torch.exp(1j * spec.angle())
        elif self.transform_type == "none":
            spec = spec
        return spec

    @property
    def stft_kwargs(self):
        return {**self.istft_kwargs, "return_complex": True}

    @property
    def istft_kwargs(self):
        return dict(
            n_fft=self.n_fft, hop_length=self.hop_length,
            window=self.window, center=True
        )

    def _get_window(self, x):
        """
        Retrieve an appropriate window for the given tensor x, matching the device.
        Caches the retrieved windows so that only one window tensor will be allocated per device.
        """
        window = self.windows.get(x.device, None)
        if window is None:
            window = self.window.to(x.device)
            self.windows[x.device] = window
        return window

    def stft(self, sig):
        window = self._get_window(sig)
        return torch.stft(sig, **{**self.stft_kwargs, "window": window})

    def istft(self, spec, length=None):
        window = self._get_window(spec)
        return torch.istft(spec, **{**self.istft_kwargs, "window": window, "length": length})

    def train_dataloader(self,contrastive_weight):
        # 使用自己的采样器，保证每个batch的每个说话人有两个
        if contrastive_weight > 0.0:
            batch_sampler = SpeakerBatchSampler(
                spk_ids=self.train_set.spk_ids,
                batch_size=self.batch_size,
                min_per_speaker=2,
                drop_last=True,
            )
            return DataLoader(
                self.train_set,
                batch_sampler=batch_sampler,
                num_workers=self.num_workers,
                pin_memory=self.gpu
            )
        else:
            return DataLoader(
                self.train_set,
                num_workers=self.num_workers,
                pin_memory=self.gpu
            )

    def val_dataloader(self):
        return DataLoader(
            self.valid_set, batch_size=self.batch_size,
            num_workers=self.num_workers, pin_memory=self.gpu, shuffle=False,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_set, batch_size=self.batch_size,
            num_workers=self.num_workers, pin_memory=self.gpu, shuffle=False,
        )

# 加载dns数据集，并加载初步增强wav
class DNS1EnhancedCondFromScp(Dataset):
    """
    Returns: X_clean, Y_main(enh), Y_cond(noisy), spk_id

    - wav_scp provides noisy wav path (uttid -> noisy path)
    - clean path is inferred by fileid: clean_dir/clean_fileid_{fileid}.wav (unless clean_scp is provided)
    - enhanced path is inferred by noisy basename: enhanced_dir/<noisy_basename>.wav
      (if missing and enhanced_fallback_to_noisy=True, use noisy as enhanced)
    """
    _re_fileid = re.compile(r"fileid_(\d+)\.wav$", re.IGNORECASE)
    _re_reader = re.compile(r"reader_(\d+)", re.IGNORECASE)
    _re_fileid_any = re.compile(r"fileid_(\d+)", re.IGNORECASE)

    def __init__(
        self,
        wav_scp: str,
        clean_dir: str,
        enhanced_dir: str,
        subset: str,
        dummy: bool,
        shuffle_spec: bool,
        num_frames: int,
        hop_length: int,
        normalize: str = "noisy",
        spec_transform: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
        stft_kwargs: Optional[dict] = None,
        clean_template: str = "clean_fileid_{fileid}.wav",
        clean_scp: Optional[str] = None,
        enhanced_fallback_to_noisy: bool = False,
        expected_sr: Optional[int] = None,
    ):
        super().__init__()
        assert stft_kwargs is not None, "stft_kwargs must be provided"
        assert normalize in ("clean", "noisy", "not")

        self.wav_items = read_kaldi_scp(wav_scp)                # utt -> noisy
        self.clean_map = scp_to_dict(read_kaldi_scp(clean_scp)) if clean_scp else None

        self.clean_dir = Path(clean_dir)
        self.enhanced_dir = Path(enhanced_dir)

        self.subset = subset
        self.dummy = dummy
        self.shuffle_spec = shuffle_spec
        self.num_frames = num_frames
        self.hop_length = hop_length
        self.normalize = normalize
        self.spec_transform = spec_transform if spec_transform is not None else (lambda x: x)
        self.stft_kwargs = stft_kwargs
        self.clean_template = clean_template
        self.enhanced_fallback_to_noisy = enhanced_fallback_to_noisy
        self.expected_sr = expected_sr

        # --------验证step 通过这个随机抽取样本 --------
        self.uttids = [utt for utt, _ in self.wav_items]
        self.noisy_files = [p for _, p in self.wav_items]
        self.spk_ids = [self._parse_spk_id(utt, p) for utt, p in self.wav_items]
        self.clean_files = [self._resolve_clean_path(utt, p) for utt, p in self.wav_items]
        self.enhanced_files = [self._resolve_enhanced_path(p) for _, p in self.wav_items]
        # ---------------------------------------------------------------
    
    def _resolve_clean_path(self, utt: str, noisy_path: str) -> str:
        if self.clean_map is not None and utt in self.clean_map:
            return self.clean_map[utt]
        fileid = self._parse_fileid(utt, noisy_path)
        clean_name = self.clean_template.format(fileid=fileid)
        return str(self.clean_dir / clean_name)

    def _resolve_enhanced_path(self, noisy_path: str) -> str:
        return str(self.enhanced_dir / Path(noisy_path).name)

    def __len__(self):
        if self.dummy:
            return max(1, int(len(self.wav_items) / 200))
        return len(self.wav_items)

    def _parse_fileid(self, utt: str, noisy_path: str) -> int:
        # Prefer noisy basename "....fileid_1861.wav"
        base = Path(noisy_path).name
        m = self._re_fileid.search(base)
        if m:
            return int(m.group(1))
        # fallback: try uttid e.g. reader_06709_fileid_1861
        m2 = self._re_fileid_any.search(utt)
        if m2:
            return int(m2.group(1))
        raise ValueError(f"Cannot parse fileid from utt={utt}, noisy={noisy_path}")

    def _parse_spk_id(self, utt: str, noisy_path: str) -> int:
        # training uttid often contains reader_06709...
        m = self._re_reader.search(utt)
        if m:
            return int(m.group(1))
        # sometimes only filename contains reader_
        m = self._re_reader.search(Path(noisy_path).name)
        if m:
            return int(m.group(1))
        return 0

    def _load_mono(self, path: str) -> torch.Tensor:
        wav, sr = load(path)   # [C, T]
        if self.expected_sr is not None and sr != self.expected_sr:
            raise ValueError(f"Sample rate mismatch for {path}: got {sr}, expected {self.expected_sr}")
        if wav.dim() == 2 and wav.size(0) > 1:
            wav = wav[:1, :]   # take first channel
        return wav

    def __getitem__(self, i):
        utt = self.uttids[i]
        noisy_path = self.noisy_files[i]
        clean_path = self.clean_files[i]
        enh_path = self.enhanced_files[i]
        spk_id = self.spk_ids[i]  
        if i < 3:
            print("utt", utt)
            print("clean", clean_path)
            print("noisy", noisy_path)
            print("enh", enh_path)
        y_noisy = self._load_mono(noisy_path)

        if os.path.exists(enh_path):
            y_pre = self._load_mono(enh_path)
        else:
            if self.enhanced_fallback_to_noisy:
                y_pre = y_noisy.clone()
            else:
                raise FileNotFoundError(f"Enhanced wav not found: {enh_path}")

        if not os.path.exists(clean_path):
            raise FileNotFoundError(f"Clean wav not found: {clean_path} (utt={utt})")
        x = self._load_mono(clean_path)

        # Align lengths by center crop/pad to target_len
        target_len = (self.num_frames - 1) * self.hop_length  # e.g. 255*128=32640
        cur_len = x.size(-1)

        # if lengths differ, align using min length first (avoid drift)
        min_len = min(x.size(-1), y_noisy.size(-1), y_pre.size(-1))
        x = x[..., :min_len]
        y_noisy = y_noisy[..., :min_len]
        y_pre = y_pre[..., :min_len]
        cur_len = min_len

        pad = max(target_len - cur_len, 0)
        if pad == 0:
            if self.shuffle_spec:
                start = int(np.random.uniform(0, cur_len - target_len))
            else:
                start = int((cur_len - target_len) / 2)
            x = x[..., start:start + target_len]
            y_noisy = y_noisy[..., start:start + target_len]
            y_pre = y_pre[..., start:start + target_len]
        else:
            left = pad // 2
            right = left + (pad % 2)
            x = F.pad(x, (left, right), mode="constant", value=0.0)
            y_noisy = F.pad(y_noisy, (left, right), mode="constant", value=0.0)
            y_pre = F.pad(y_pre, (left, right), mode="constant", value=0.0)

        # 归一化(keep consistent scaling across the three)
        if self.normalize == "noisy":
            normfac = y_noisy.abs().max()
        elif self.normalize == "clean":
            normfac = x.abs().max()
        else:
            normfac = torch.tensor(1.0, dtype=x.dtype, device=x.device)
        normfac = normfac.clamp_min(1e-12)

        x = x / normfac
        y_noisy = y_noisy / normfac
        y_pre = y_pre / normfac

        # STFT -> spec
        X = torch.stft(x, **self.stft_kwargs)
        Y_noisy = torch.stft(y_noisy, **self.stft_kwargs)
        Y_pre = torch.stft(y_pre, **self.stft_kwargs)

        X = self.spec_transform(X)
        Y_noisy = self.spec_transform(Y_noisy)
        Y_pre = self.spec_transform(Y_pre)

        return X, Y_noisy,Y_pre, spk_id

class DNS1EnhancedCondScpDataModule(pl.LightningDataModule):
    """
    联合判别式模型的训练，从scp文件加载DSN1数据集。样本除noisy、clean，还有初步增强的wav。
    届时self.train_set、self.valid_set将存储训练集和验证集。
    并且self.train_set中有noisy_files clean_files enhanced_files 分别放原始含噪语音、干净语音、初步增强语音。
    Loads train/valid/test by scp:
      train: train_wav_scp
      valid: valid_wav_scp
      test : test_wav_scp (optional), and test_clean_scp (optional)
    """
    @staticmethod
    def add_argparse_args(parser):
        # scp
        parser.add_argument("--train_wav_scp", type=str, required=True)
        parser.add_argument("--valid_wav_scp", type=str, required=True)
        parser.add_argument("--test_wav_scp", type=str, default=None)
        parser.add_argument("--test_clean_scp", type=str, default=None)

        # dirs
        parser.add_argument("--clean_dir", type=str, required=True)
        parser.add_argument("--enhanced_dir", type=str, required=True)

        # audio/stft
        parser.add_argument("--batch_size", type=int, default=8)
        parser.add_argument("--n_fft", type=int, default=510)
        parser.add_argument("--hop_length", type=int, default=128)
        parser.add_argument("--num_frames", type=int, default=256)
        parser.add_argument("--window", type=str, choices=("sqrthann", "hann"), default="hann")
        parser.add_argument("--num_workers", type=int, default=4)
        parser.add_argument("--dummy", action="store_true")

        # spec transform
        parser.add_argument("--spec_factor", type=float, default=0.15)
        parser.add_argument("--spec_abs_exponent", type=float, default=0.5)
        parser.add_argument("--normalize", type=str, choices=("clean", "noisy", "not"), default="noisy")
        parser.add_argument("--transform_type", type=str, choices=("exponent", "log", "none"), default="exponent")

        # optional behavior
        parser.add_argument("--enhanced_fallback_to_noisy", action="store_true")
        parser.add_argument("--expected_sr", type=int, default=None)

        return parser

    def __init__(
        self,
        train_wav_scp: str,
        valid_wav_scp: str,
        clean_dir: str,
        enhanced_dir: str,
        test_wav_scp: Optional[str] = None,
        test_clean_scp: Optional[str] = None,
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
        enhanced_fallback_to_noisy: bool = False,
        expected_sr: Optional[int] = None,
        **kwargs,
    ):
        super().__init__()
        self.train_wav_scp = train_wav_scp
        self.valid_wav_scp = valid_wav_scp
        self.test_wav_scp = test_wav_scp
        self.test_clean_scp = test_clean_scp

        self.clean_dir = clean_dir
        self.enhanced_dir = enhanced_dir

        self.batch_size = batch_size
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.num_frames = num_frames
        self.window = get_window(window, self.n_fft)
        self.windows = {}
        self.num_workers = num_workers
        self.dummy = dummy

        self.spec_factor = spec_factor
        self.spec_abs_exponent = spec_abs_exponent
        self.normalize = normalize
        self.transform_type = transform_type

        self.enhanced_fallback_to_noisy = enhanced_fallback_to_noisy
        self.expected_sr = expected_sr

        self.kwargs = kwargs

    @property
    def stft_kwargs(self):
        return {**self.istft_kwargs, "return_complex": True}
    @property
    def istft_kwargs(self):
        return dict(
            n_fft=self.n_fft, hop_length=self.hop_length,
            window=self.window, center=True
        )
    def spec_fwd(self, spec: torch.Tensor) -> torch.Tensor:
        if self.transform_type == "exponent":
            if self.spec_abs_exponent != 1:
                e = self.spec_abs_exponent
                spec = spec.abs() ** e * torch.exp(1j * spec.angle())
            spec = spec * self.spec_factor
        elif self.transform_type == "log":
            spec = torch.log(1 + spec.abs()) * torch.exp(1j * spec.angle())
            spec = spec * self.spec_factor
        elif self.transform_type == "none":
            pass
        return spec

    def setup(self, stage=None):
        ds_kwargs = dict(
            clean_dir=self.clean_dir,
            enhanced_dir=self.enhanced_dir,
            num_frames=self.num_frames,
            hop_length=self.hop_length,
            normalize=self.normalize,
            spec_transform=self.spec_fwd,
            stft_kwargs=self.stft_kwargs,
            enhanced_fallback_to_noisy=self.enhanced_fallback_to_noisy,
            expected_sr=self.expected_sr,
        )

        if stage in ("fit", None):
            self.train_set = DNS1EnhancedCondFromScp(
                wav_scp=self.train_wav_scp,
                subset="train",
                dummy=self.dummy,
                shuffle_spec=True,
                **ds_kwargs,
            )
            self.valid_set = DNS1EnhancedCondFromScp(
                wav_scp=self.valid_wav_scp,
                subset="valid",
                dummy=self.dummy,
                shuffle_spec=False,
                **ds_kwargs,
            )
        if stage in ("test", None):
            if self.test_wav_scp is None:
                self.test_set = None
            else:
                self.test_set = DNS1EnhancedCondFromScp(
                    wav_scp=self.test_wav_scp,
                    clean_scp=self.test_clean_scp,
                    subset="test",
                    dummy=self.dummy,
                    shuffle_spec=False,
                    **ds_kwargs,
                )

    def train_dataloader(self, contrastive_weight):
         # 对比学习使用自定义采样器，保证每个batch的每个说话人有两个
        if contrastive_weight > 0.0:
            batch_sampler = SpeakerBatchSampler(
                spk_ids=self.train_set.spk_ids,
                batch_size=self.batch_size,
                min_per_speaker=2,
                drop_last=True,
            )
            return DataLoader(
                self.train_set,
                batch_sampler=batch_sampler,
                num_workers=self.num_workers,
                pin_memory=True
            )
        else:
            return DataLoader(
                self.train_set,
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=self.num_workers,
                pin_memory=True,
                drop_last=True,
            )
       

    def val_dataloader(self):
        return DataLoader(
            self.valid_set,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=False,
        )

    def test_dataloader(self):
        if self.test_set is None:
            return None
        return DataLoader(
            self.test_set,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=False,
        )
    def spec_back(self, spec):
        if self.transform_type == "exponent":
            spec = spec / self.spec_factor
            if self.spec_abs_exponent != 1:
                e = self.spec_abs_exponent
                spec = spec.abs()**(1/e) * torch.exp(1j * spec.angle())
        elif self.transform_type == "log":
            spec = spec / self.spec_factor
            spec = (torch.exp(spec.abs()) - 1) * torch.exp(1j * spec.angle())
        elif self.transform_type == "none":
            spec = spec
        return spec
    def _get_window(self, x):
        """
        Retrieve an appropriate window for the given tensor x, matching the device.
        Caches the retrieved windows so that only one window tensor will be allocated per device.
        """
        window = self.windows.get(x.device, None)
        if window is None:
            window = self.window.to(x.device)
            self.windows[x.device] = window
        return window

    def stft(self, sig):
        window = self._get_window(sig)
        return torch.stft(sig, **{**self.stft_kwargs, "window": window})

    def istft(self, spec, length=None):
        window = self._get_window(spec)
        return torch.istft(spec, **{**self.istft_kwargs, "window": window, "length": length})


#仅加载dns数据集 
class DNS1FromScp(Dataset):
    """
    Returns: X_clean, Y_noisy, spk_id

    - wav_scp provides noisy wav path (uttid -> noisy path)
    - clean path is inferred by fileid: clean_dir/clean_fileid_{fileid}.wav (unless clean_scp is provided)
    """
    _re_fileid = re.compile(r"fileid_(\d+)\.wav$", re.IGNORECASE)
    _re_reader = re.compile(r"reader_(\d+)", re.IGNORECASE)
    _re_fileid_any = re.compile(r"fileid_(\d+)", re.IGNORECASE)

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
        spec_transform: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
        stft_kwargs: Optional[dict] = None,
        clean_template: str = "clean_fileid_{fileid}.wav",
        clean_scp: Optional[str] = None,
        expected_sr: Optional[int] = None,
        **ignored_kwargs,   
    ):
        super().__init__()
        assert stft_kwargs is not None, "stft_kwargs must be provided"
        assert normalize in ("clean", "noisy", "not")
        assert all(k in stft_kwargs for k in ["n_fft", "hop_length", "center", "window"]), "misconfigured STFT kwargs"
        assert stft_kwargs.get("center", None) is True, "'center' must be True for current implementation"

        self.wav_items = read_kaldi_scp(wav_scp)                # utt -> noisy
        self.clean_map = scp_to_dict(read_kaldi_scp(clean_scp)) if clean_scp else None

        self.clean_dir = Path(clean_dir)
        self.subset = subset
        self.dummy = dummy
        self.shuffle_spec = shuffle_spec
        self.num_frames = num_frames
        self.hop_length = hop_length
        self.normalize = normalize
        self.spec_transform = spec_transform if spec_transform is not None else (lambda x: x)
        self.stft_kwargs = stft_kwargs
        self.clean_template = clean_template
        self.expected_sr = expected_sr

        # 用于验证抽样/可视化/采样器
        self.uttids = [utt for utt, _ in self.wav_items]
        self.noisy_files = [p for _, p in self.wav_items]
        self.clean_files = [self._resolve_clean_path(utt, p) for utt, p in self.wav_items]
        self.spk_ids = [self._parse_spk_id(utt, p) for utt, p in self.wav_items]

    def _resolve_clean_path(self, utt: str, noisy_path: str) -> str:
        if self.clean_map is not None and utt in self.clean_map:
            return self.clean_map[utt]
        fileid = self._parse_fileid(utt, noisy_path)
        clean_name = self.clean_template.format(fileid=fileid)
        return str(self.clean_dir / clean_name)

    def __len__(self):
        if self.dummy:
            return max(1, int(len(self.wav_items) / 200))
        return len(self.wav_items)

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

    def _load_mono(self, path: str) -> torch.Tensor:
        wav, sr = load(path)   # [C, T]
        if self.expected_sr is not None and sr != self.expected_sr:
            raise ValueError(f"Sample rate mismatch for {path}: got {sr}, expected {self.expected_sr}")
        if wav.dim() == 2 and wav.size(0) > 1:
            wav = wav[:1, :]
        return wav

    def __getitem__(self, i):
        utt = self.uttids[i]
        noisy_path = self.noisy_files[i]
        clean_path = self.clean_files[i]
        spk_id = self.spk_ids[i]

        # noisy waveform
        y_noisy = self._load_mono(noisy_path)

        # clean waveform
        if not os.path.exists(clean_path):
            raise FileNotFoundError(f"Clean wav not found: {clean_path} (utt={utt})")
        x = self._load_mono(clean_path)

        # 对齐长度（避免 drift）
        min_len = min(x.size(-1), y_noisy.size(-1))
        x = x[..., :min_len]
        y_noisy = y_noisy[..., :min_len]
        cur_len = min_len

        # center=True 下 target_len 公式
        target_len = (self.num_frames - 1) * self.hop_length
        pad = max(target_len - cur_len, 0)

        if pad == 0:
            if self.shuffle_spec:
                start = int(np.random.uniform(0, cur_len - target_len))
            else:
                start = int((cur_len - target_len) / 2)
            x = x[..., start:start + target_len]
            y_noisy = y_noisy[..., start:start + target_len]
        else:
            left = pad // 2
            right = left + (pad % 2)
            x = F.pad(x, (left, right), mode="constant", value=0.0)
            y_noisy = F.pad(y_noisy, (left, right), mode="constant", value=0.0)

        # normalize（保证 x/y 同尺度）
        if self.normalize == "noisy":
            normfac = y_noisy.abs().max()
        elif self.normalize == "clean":
            normfac = x.abs().max()
        else:  # "not"
            normfac = torch.tensor(1.0, dtype=x.dtype, device=x.device)
        normfac = normfac.clamp_min(1e-12)

        x = x / normfac
        y_noisy = y_noisy / normfac

        # STFT -> spec
        X = torch.stft(x, **self.stft_kwargs)
        Y_noisy = torch.stft(y_noisy, **self.stft_kwargs)

        X = self.spec_transform(X)
        Y_noisy = self.spec_transform(Y_noisy)

        return X, Y_noisy, spk_id

class DNS1ScpDataModule(pl.LightningDataModule):
    """
    仅加载 DNS noisy/clean（scp方式），不加载 enhanced wav
    """
    @staticmethod
    def add_argparse_args(parser):
        parser.add_argument("--train_wav_scp", type=str, required=True)
        parser.add_argument("--valid_wav_scp", type=str, required=True)
        parser.add_argument("--test_wav_scp", type=str, default=None)
        parser.add_argument("--test_clean_scp", type=str, default=None)

        parser.add_argument("--clean_dir", type=str, required=True)

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
        parser.add_argument("--clean_template", type=str, default="clean_fileid_{fileid}.wav")
        return parser

    def __init__(
        self,
        train_wav_scp: str,
        valid_wav_scp: str,
        clean_dir: str,
        test_wav_scp: Optional[str] = None,
        test_clean_scp: Optional[str] = None,
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
        expected_sr: Optional[int] = None,
        clean_template: str = "clean_fileid_{fileid}.wav",
        gpu: bool = True,
        **kwargs,
    ):
        super().__init__()
        self.train_wav_scp = train_wav_scp
        self.valid_wav_scp = valid_wav_scp
        self.test_wav_scp = test_wav_scp
        self.test_clean_scp = test_clean_scp

        self.clean_dir = clean_dir
        self.clean_template = clean_template

        self.batch_size = batch_size
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.num_frames = num_frames
        self.window = get_window(window, self.n_fft)
        self.windows = {}
        self.num_workers = num_workers
        self.dummy = dummy
        self.gpu = gpu

        self.spec_factor = spec_factor
        self.spec_abs_exponent = spec_abs_exponent
        self.normalize = normalize
        self.transform_type = transform_type

        self.expected_sr = expected_sr
        self.kwargs = kwargs  # ✅ 你项目里风格：保留，但 Dataset 会忽略多余项

    @property
    def stft_kwargs(self):
        return {**self.istft_kwargs, "return_complex": True}

    @property
    def istft_kwargs(self):
        return dict(
            n_fft=self.n_fft, hop_length=self.hop_length,
            window=self.window, center=True
        )

    def spec_fwd(self, spec: torch.Tensor) -> torch.Tensor:
        if self.transform_type == "exponent":
            if self.spec_abs_exponent != 1:
                e = self.spec_abs_exponent
                spec = spec.abs() ** e * torch.exp(1j * spec.angle())
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
        return spec

    def setup(self, stage=None):
        ds_kwargs = dict(
            clean_dir=self.clean_dir,
            num_frames=self.num_frames,
            hop_length=self.hop_length,
            normalize=self.normalize,
            spec_transform=self.spec_fwd,
            stft_kwargs=self.stft_kwargs,
            expected_sr=self.expected_sr,
            clean_template=self.clean_template
        )

        if stage in ("fit", None):
            self.train_set = DNS1FromScp(
                wav_scp=self.train_wav_scp,
                subset="train",
                dummy=self.dummy,
                shuffle_spec=True,
                **ds_kwargs,
            )
            self.valid_set = DNS1FromScp(
                wav_scp=self.valid_wav_scp,
                subset="valid",
                dummy=self.dummy,
                shuffle_spec=False,
                **ds_kwargs,
            )

        if stage in ("test", None):
            if self.test_wav_scp is None:
                self.test_set = None
            else:
                self.test_set = DNS1FromScp(
                    wav_scp=self.test_wav_scp,
                    clean_scp=self.test_clean_scp,
                    subset="test",
                    dummy=self.dummy,
                    shuffle_spec=False,
                    **ds_kwargs,
                )

    def train_dataloader(self, contrastive_weight):
        if contrastive_weight > 0.0:
            batch_sampler = SpeakerBatchSampler(
                spk_ids=self.train_set.spk_ids,
                batch_size=self.batch_size,
                min_per_speaker=2,
                drop_last=True,
            )
            return DataLoader(
                self.train_set,
                batch_sampler=batch_sampler,
                num_workers=self.num_workers,
                pin_memory=self.gpu
            )
        else:
            return DataLoader(
                self.train_set,
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=self.num_workers,
                pin_memory=self.gpu,
                drop_last=True,
            )

    def val_dataloader(self):
        return DataLoader(
            self.valid_set,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.gpu,
            drop_last=False,
        )

    def test_dataloader(self):
        if self.test_set is None:
            return None
        return DataLoader(
            self.test_set,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.gpu,
            drop_last=False,
        )
    def _get_window(self, x):
        """
        Retrieve an appropriate window for the given tensor x, matching the device.
        Caches the retrieved windows so that only one window tensor will be allocated per device.
        """
        window = self.windows.get(x.device, None)
        if window is None:
            window = self.window.to(x.device)
            self.windows[x.device] = window
        return window

    def stft(self, sig):
        window = self._get_window(sig)
        return torch.stft(sig, **{**self.stft_kwargs, "window": window})

    def istft(self, spec, length=None):
        window = self._get_window(spec)
        return torch.istft(spec, **{**self.istft_kwargs, "window": window, "length": length})

