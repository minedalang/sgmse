# coding=utf-8
# Copyright 2020 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# pylint: skip-file
'''
NCSN++ v2 backbone with:
- bottleneck cross-attn using q=h, k/v=cond
- optional decoder FiLM on selected levels
- optional lightweight TF-Grid-style evidence selector that builds Y_hint from (y, y_cond) 
    加入轻量化TF-GridNet作为evidence selector，从y_cond中预测y_hint通过交叉注意力注入网络主干
    TF-GridNet有A，B两个方案，A是输出一个soft mask, 作为noisy的掩码，指明noisy像语音的部分。B是输出softmask M和置信度C，表示noisy的哪里像语音，并指明有多信。
- low-resolution cross-attn blocks that inject Y_hint instead of raw noisy FiLM

参考训练命令 CUDA_VISIBLE_DEVICES=4,5,6,7 python joint_train.py --sde sbve --loss_type denoiser --loss_weighting edm --c_in edm --c_out edm --c_skip edm --cross_attn  --train_type mode2 --selector_type tfgrid_a --selector_hidden_ch 32 --selector_num_blocks 2 --selector_attn_levels 6,5 --dataset joint_scp --train_wav_scp /home/xyf/cwh_project/DNS-Challenge-1/dns1-repo/synthetic_snr-20_-15/scp/tr_synthetic/wav.scp --valid_wav_scp /home/xyf/cwh_project/DNS-Challenge-1/dns1-repo/synthetic_snr-20_-15/scp/cv_synthetic/wav.scp --test_wav_scp /home/xyf/cwh_project/DNS-Challenge-1/dns1-repo/synthetic_snr-20_-15/scp/tt_real_recordings/wav.scp --clean_dir /home/xyf/cwh_project/DNS-Challenge-1/dns1-repo/synthetic_snr-20_-15/clean --batch_size 8 --wandb_name sgmse-sbve-edm-nofilm-tfgridnetA-model2-bs4*8-joint-freezeSemmaba-noTrainedSgmse --semamba_ckpt /home/xyf/cwh_project/SEMamba-main/SEMamba-main/exp/SEMamba_trainedby_dnslowsnr-20_-15/g_00134000.pth --semamba_root /home/xyf/cwh_project/SEMamba-main/SEMamba-main --semamba_config /home/xyf/cwh_project/SEMamba-main/SEMamba-main/exp/SEMamba_trainedby_dnslowsnr-20_-15/config.yaml
'''
from .ncsnpp_utils import layers, layerspp, normalization
import torch.nn as nn
import functools
import torch
import numpy as np
import torch.nn.functional as F

from .shared import BackboneRegistry

ResnetBlockDDPM = layerspp.ResnetBlockDDPMpp
ResnetBlockBigGAN = layerspp.ResnetBlockBigGANpp
Combine = layerspp.Combine
conv3x3 = layerspp.conv3x3
conv1x1 = layerspp.conv1x1
get_act = layers.get_act
get_normalization = normalization.get_normalization
default_initializer = layers.default_init

def _parse_level_spec(levels_str: str, default: str = "none"):
    """Parse level string into a set of decoder level indices.

    Decoder loop uses i_level in reversed(range(num_resolutions)); larger i_level means lower resolution.
    Examples: 'all', 'none', 'first', 'last', '6,5'.
    """
    if levels_str is None:
        levels_str = default
    levels_str = str(levels_str).lower().strip()
    if levels_str == "none":
        return set()
    if levels_str == "all":
        return set(range(32))
    if levels_str == "first":
        return {0}
    if levels_str == "last":
        return {6}
    levels = set()
    for part in levels_str.split(','):
        part = part.strip()
        if part:
            levels.add(int(part))
    return levels


def _complex_to_2ch(y: torch.Tensor) -> torch.Tensor:
    return torch.cat([y.real, y.imag], dim=1)


class BottleneckCrossAttention(nn.Module):
    """
    在 U-Net 瓶颈处，用原始含噪语音 y 对主干特征 h 做一次 cross-attention：
      - h: (B, C, H, W)  主干特征（瓶颈）
      - y: (B, 1, H_y, W_y)  复数谱（complex tensor）
    - 先把 y -> cond(y) （非线性卷积），作为 Q 的输入；
    - h 归一化后生成 K/V；
    - Attention:  - q 来自主干 h  - k/v 来自条件 y_cond 主干特征h去条件里找“我现在最该参考什么语音信息”。
    再把得到的表示映射回 h 的空间上，并 residual 加到 h 上。
    """
    def __init__(self, channels: int, cond_in_channels: int = 2):
        super().__init__()
        self.channels = channels

        self.norm = nn.GroupNorm(
            num_groups=min(channels // 4, 32),
            num_channels=channels,
            eps=1e-6
        )

        # 条件编码稍微做深一点
        self.cond_net = nn.Sequential(
            nn.Conv2d(cond_in_channels, channels, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.SiLU(),
        )

        self.q_proj = nn.Conv2d(channels, channels, kernel_size=1)
        self.k_proj = nn.Conv2d(channels, channels, kernel_size=1)
        self.v_proj = nn.Conv2d(channels, channels, kernel_size=1)
        self.proj_out = nn.Conv2d(channels, channels, kernel_size=1)

        # 条件注入强度
        self.gate = nn.Parameter(torch.tensor(0.1))

        for m in [
            self.cond_net[0], self.cond_net[2],
            self.q_proj, self.k_proj, self.v_proj, self.proj_out
        ]:
            m.weight.data = default_initializer()(m.weight.shape)
            nn.init.zeros_(m.bias)

    def forward(self, h: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        h: (B, C, H, W)
        y: (B, 1, H_y, W_y) complex tensor
        """
        B, C, H, W = h.shape

        cond = torch.cat([y.real, y.imag], dim=1)  # (B, 2, H_y, W_y)
        if cond.shape[2] != H or cond.shape[3] != W:
            cond = F.interpolate(cond, size=(H, W), mode="bilinear", align_corners=False)

        cond = self.cond_net(cond)      # (B, C, H, W)
        h_norm = self.norm(h)

        # 关键改动：q from h, k/v from cond
        q = self.q_proj(h_norm)
        k = self.k_proj(cond)
        v = self.v_proj(cond)

        q = q.view(B, C, H * W).transpose(1, 2)  # (B, N, C)
        k = k.view(B, C, H * W).transpose(1, 2)  # (B, N, C)
        v = v.view(B, C, H * W).transpose(1, 2)  # (B, N, C)

        scale = 1.0 / np.sqrt(C)
        attn = torch.softmax(torch.bmm(q, k.transpose(1, 2)) * scale, dim=-1)
        out = torch.bmm(attn, v)

        out = out.transpose(1, 2).view(B, C, H, W)
        out = self.proj_out(out)

        return h + self.gate * out
    
class TFGridLiteBlock(nn.Module):
    """A lightweight TF-grid-style block: separate frequency/time depthwise mixing + residual gate."""
    def __init__(self, channels: int):
        super().__init__()
        groups = max(1, min(channels // 4, 32))
        self.norm = nn.GroupNorm(groups, channels, eps=1e-6)
        self.in_proj = nn.Conv2d(channels, channels, kernel_size=1)
        self.freq_dw = nn.Conv2d(channels, channels, kernel_size=(3, 1), padding=(1, 0), groups=channels)
        self.time_dw = nn.Conv2d(channels, channels, kernel_size=(1, 3), padding=(0, 1), groups=channels)
        self.mix = nn.Sequential(
            nn.SiLU(),
            nn.Conv2d(channels, channels, kernel_size=1),
            nn.SiLU(),
            nn.Conv2d(channels, channels, kernel_size=1),
        )
        self.gate = nn.Parameter(torch.tensor(0.1))
        for m in [self.in_proj, self.freq_dw, self.time_dw, self.mix[1], self.mix[3]]:
            m.weight.data = default_initializer()(m.weight.shape)
            nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.in_proj(self.norm(x))
        h = self.freq_dw(h) + self.time_dw(h)
        h = self.mix(h)
        return x + self.gate * h


class TFGridEvidenceSelector(nn.Module):
    """Lightweight selector that converts (y_main, y_aux) into a speech-biased Y_hint.

    Version A: Y_hint = M * y_aux
    Version B: Y_hint = (M * C) * y_aux where C is an extra confidence branch.
    """
    def __init__(self, in_channels: int = 6, hidden_channels: int = 32, num_blocks: int = 2, version: str = "a"):
        super().__init__()
        self.version = version.lower()
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, kernel_size=3, padding=1),
            nn.SiLU(),
        )
        self.blocks = nn.Sequential(*[TFGridLiteBlock(hidden_channels) for _ in range(num_blocks)])
        self.mask_head = nn.Conv2d(hidden_channels, 1, kernel_size=1)
        self.conf_head = nn.Conv2d(hidden_channels, 1, kernel_size=1) if self.version == "b" else None
        init_layers = [self.stem[0], self.mask_head]
        if self.conf_head is not None:
            init_layers.append(self.conf_head)
        for m in init_layers:
            m.weight.data = default_initializer()(m.weight.shape)
            nn.init.zeros_(m.bias)

    def forward(self, y_main: torch.Tensor, y_aux: torch.Tensor, return_mask: bool = False):
        diff = y_aux - y_main
        feat = torch.cat([
            y_main.real, y_main.imag,
            y_aux.real, y_aux.imag,
            diff.real, diff.imag,
        ], dim=1)
        h = self.blocks(self.stem(feat))
        mask = torch.sigmoid(self.mask_head(h))
        if self.conf_head is not None:
            conf = torch.sigmoid(self.conf_head(h))
            scale = mask * conf
        else:
            conf = None
            scale = mask
        y_hint = torch.complex(y_aux.real * scale, y_aux.imag * scale)
        if return_mask:
            return y_hint, mask, conf
        return y_hint


class DecoderResBlockWithFiLM(nn.Module):
    """
    在 decoder 的每个 ResBlock 后做一次 FiLM / AdaGN 风格调制，
    比“每个 level 末尾调一次”更强、更直接。
    """
    @staticmethod
    def _parse_film_levels(film_levels_str: str):
        return _parse_level_spec(film_levels_str, default="first")

    def __init__(
        self,
        resblock_ctor,
        in_ch: int,
        out_ch: int,
        cond_in_channels: int = 2,
        cond_hidden_ch: int = None,
        init_scale: float = 0.0,
    ):
        super().__init__()
        self.block = resblock_ctor(in_ch=in_ch, out_ch=out_ch)

        cond_hidden_ch = out_ch if cond_hidden_ch is None else cond_hidden_ch

        self.norm = nn.GroupNorm(
            num_groups=min(out_ch // 4, 32),
            num_channels=out_ch,
            eps=1e-6
        )

        self.cond_net = nn.Sequential(
            nn.Conv2d(cond_in_channels, cond_hidden_ch, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Conv2d(cond_hidden_ch, 2 * out_ch, kernel_size=3, padding=1),
        )

        self.proj_out = conv3x3(out_ch, out_ch, init_scale=init_scale)

        # 条件调制强度
        self.gate = nn.Parameter(torch.tensor(0.1))

        for m in [self.cond_net[0], self.cond_net[2], self.proj_out]:
            m.weight.data = default_initializer()(m.weight.shape)
            nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor, temb: torch.Tensor, y_cond: torch.Tensor) -> torch.Tensor:
        h = self.block(x, temb)  # 先过原始 ResBlock

        B, C, H, W = h.shape
        cond = torch.cat([y_cond.real, y_cond.imag], dim=1)

        if cond.shape[2] != H or cond.shape[3] != W:
            cond = F.interpolate(cond, size=(H, W), mode="bilinear", align_corners=False)

        gamma, beta = torch.chunk(self.cond_net(cond), 2, dim=1)

        # AdaGN / FiLM 风格
        h_mod = self.norm(h)
        h_mod = h_mod * (1.0 + gamma) + beta
        h_mod = F.silu(h_mod)
        h_mod = self.proj_out(h_mod)

        return h + self.gate * h_mod
    
@BackboneRegistry.register("ncsnpp_v2")
class NCSNpp_v2(nn.Module):

    @staticmethod
    def _parse_film_levels(film_levels_str: str):
        return _parse_level_spec(film_levels_str, default="first")

    """NCSN++ model, adapted from https://github.com/yang-song/score_sde repository"""

    @staticmethod
    def add_argparse_args(parser):
        parser.add_argument("--nf", type=int, default=128)
        parser.add_argument("--ch_mult",type=int, nargs='+', default=[1,1,2,2,2,2,2])
        parser.add_argument("--num_res_blocks", type=int, default=2)
        parser.add_argument("--attn_resolutions", type=int, nargs='+', default=[16])
        parser.add_argument("--cross_attn", action='store_true', default=False)
        parser.add_argument("--film_modulate", action='store_true', default=False)
        parser.add_argument("--film_levels", type=str, default="first",
            help="FiLM applied to which upsample levels: 'all', 'first', 'last', 'none', or comma-separated indices like '0,1,2'")
        parser.add_argument("--selector_type", type=str, default="none", choices=("none", "tfgrid_a", "tfgrid_b"),
            help="Evidence selector type. tfgrid_a: mask only, tfgrid_b: mask * confidence.")
        parser.add_argument("--selector_hidden_ch", type=int, default=32, help="Hidden channels in lightweight TF-grid selector.")
        parser.add_argument("--selector_num_blocks", type=int, default=2, help="Number of TF-grid-style blocks in selector.")
        parser.add_argument("--selector_attn_levels", type=str, default="6,5",
            help="Decoder low-resolution levels that receive Y_hint via cross-attn, e.g. '6,5'.")
        return parser
    def __init__(self,
        nf = 128,
        ch_mult = (1, 1, 2, 2, 2, 2, 2),
        num_res_blocks = 2,
        attn_resolutions = (16,),
        cross_attn = False,
        film_modulate=False,
        film_levels="first",
        selector_type="none",
        selector_hidden_ch=32,
        selector_num_blocks=2,
        selector_attn_levels="6,5",
        nonlinearity = 'swish',
        resamp_with_conv = True,
        fir = True,
        fir_kernel = [1, 3, 3, 1],
        skip_rescale = True,
        resblock_type = 'biggan',
        progressive = 'output_skip',
        progressive_input = 'input_skip',
        progressive_combine = 'sum',
        init_scale = 0.,
        fourier_scale = 16,
        image_size = 256,
        embedding_type = 'fourier',
        dropout = .0,
        **unused_kwargs
    ):
        super().__init__()
        self.act = act = get_act(nonlinearity)

        self.nf = nf = nf
        ch_mult = ch_mult
        self.num_res_blocks = num_res_blocks = num_res_blocks
        self.attn_resolutions = attn_resolutions = attn_resolutions
        self.cross_attn = cross_attn
        self.film_modulate = film_modulate
        self.film_levels = self._parse_film_levels(film_levels)
        self.selector_type = str(selector_type).lower()
        self.use_selector = self.selector_type != "none"
        self.selector_attn_levels = _parse_level_spec(selector_attn_levels, default="6,5")
        self.num_resolutions = num_resolutions = len(ch_mult)
        self.all_resolutions = all_resolutions = [image_size // (2 ** i) for i in range(num_resolutions)]
        self.skip_rescale = skip_rescale = skip_rescale
        self.resblock_type = resblock_type = resblock_type.lower()
        self.progressive = progressive = progressive.lower()
        self.progressive_input = progressive_input = progressive_input.lower()
        self.embedding_type = embedding_type = embedding_type.lower()

        assert progressive in ['none', 'output_skip', 'residual']
        assert progressive_input in ['none', 'input_skip', 'residual']
        assert embedding_type in ['fourier', 'positional']
        combine_method = progressive_combine.lower()
        combiner = functools.partial(Combine, method=combine_method)

        in_channels = 4   # x.real, x.imag, y.real, y.imag
        out_channels = 2  # score.real, score.imag
        self.output_layer = nn.Conv2d(in_channels, out_channels, 1)

        modules = []
        # timestep/noise_level embedding
        if embedding_type == 'fourier':
            # Gaussian Fourier features embeddings.
            modules.append(layerspp.GaussianFourierProjection(
                embedding_size=nf, scale=fourier_scale
            ))
            embed_dim = 2 * nf
        elif embedding_type == 'positional':
            embed_dim = nf
        else:
            raise ValueError(f'embedding type {embedding_type} unknown.')

        modules.append(nn.Linear(embed_dim, nf * 4))
        modules[-1].weight.data = default_initializer()(modules[-1].weight.shape)
        nn.init.zeros_(modules[-1].bias)
        modules.append(nn.Linear(nf * 4, nf * 4))
        modules[-1].weight.data = default_initializer()(modules[-1].weight.shape)
        nn.init.zeros_(modules[-1].bias)

        AttnBlock = functools.partial(layerspp.AttnBlockpp,
            init_scale=init_scale, skip_rescale=skip_rescale)

        Upsample = functools.partial(layerspp.Upsample,
            with_conv=resamp_with_conv, fir=fir, fir_kernel=fir_kernel)

        if progressive == 'output_skip':
            self.pyramid_upsample = layerspp.Upsample(fir=fir, fir_kernel=fir_kernel, with_conv=False)
        elif progressive == 'residual':
            pyramid_upsample = functools.partial(layerspp.Upsample, fir=fir,
                fir_kernel=fir_kernel, with_conv=True)

        Downsample = functools.partial(layerspp.Downsample, with_conv=resamp_with_conv, fir=fir, fir_kernel=fir_kernel)

        if progressive_input == 'input_skip':
            self.pyramid_downsample = layerspp.Downsample(fir=fir, fir_kernel=fir_kernel, with_conv=False)
        elif progressive_input == 'residual':
            pyramid_downsample = functools.partial(layerspp.Downsample,
                fir=fir, fir_kernel=fir_kernel, with_conv=True)

        if resblock_type == 'ddpm':
            ResnetBlock = functools.partial(ResnetBlockDDPM, act=act,
                dropout=dropout, init_scale=init_scale,
                skip_rescale=skip_rescale, temb_dim=nf * 4)

        elif resblock_type == 'biggan':
            ResnetBlock = functools.partial(ResnetBlockBigGAN, act=act,
                dropout=dropout, fir=fir, fir_kernel=fir_kernel,
                init_scale=init_scale, skip_rescale=skip_rescale, temb_dim=nf * 4)

        else:
            raise ValueError(f'resblock type {resblock_type} unrecognized.')

        if self.use_selector:
            selector_version = "a" if self.selector_type == "tfgrid_a" else "b"
            self.evidence_selector = TFGridEvidenceSelector(
                in_channels=6,
                hidden_channels=int(selector_hidden_ch),
                num_blocks=int(selector_num_blocks),
                version=selector_version,
            )
        else:
            self.evidence_selector = None

        # Downsampling block

        channels = in_channels
        if progressive_input != 'none':
            input_pyramid_ch = channels

        modules.append(conv3x3(channels, nf))
        hs_c = [nf]

        in_ch = nf
        for i_level in range(num_resolutions):
            # Residual blocks for this resolution
            for i_block in range(num_res_blocks):
                out_ch = nf * ch_mult[i_level]
                modules.append(ResnetBlock(in_ch=in_ch, out_ch=out_ch))
                in_ch = out_ch

                if all_resolutions[i_level] in attn_resolutions:
                    modules.append(AttnBlock(channels=in_ch))
                hs_c.append(in_ch)

            if i_level != num_resolutions - 1:
                if resblock_type == 'ddpm':
                    modules.append(Downsample(in_ch=in_ch))
                else:
                    modules.append(ResnetBlock(down=True, in_ch=in_ch))

                if progressive_input == 'input_skip':
                    modules.append(combiner(dim1=input_pyramid_ch, dim2=in_ch))
                    if combine_method == 'cat':
                        in_ch *= 2

                elif progressive_input == 'residual':
                    modules.append(pyramid_downsample(in_ch=input_pyramid_ch, out_ch=in_ch))
                    input_pyramid_ch = in_ch

                hs_c.append(in_ch)

        in_ch = hs_c[-1]
        if self.cross_attn:
            # 瓶颈处的 cross-attention
            self.bottleneck_cross_attn = BottleneckCrossAttention(
                channels=in_ch,
                cond_in_channels=2  # y.real + y.imag
            )

        modules.append(ResnetBlock(in_ch=in_ch))
        modules.append(AttnBlock(channels=in_ch))
        modules.append(ResnetBlock(in_ch=in_ch))

        pyramid_ch = 0
        # Upsampling block 上采样
        for i_level in reversed(range(num_resolutions)):
            for i_block in range(num_res_blocks + 1):
                out_ch = nf * ch_mult[i_level]
                skip_ch = hs_c.pop()
                # 只在i_level=0层级的上采样块加film
                if self.film_modulate and i_level in self.film_levels:
                    modules.append(
                        DecoderResBlockWithFiLM(   #增加了film的上采样块
                            resblock_ctor=ResnetBlock,
                            in_ch=in_ch + skip_ch,
                            out_ch=out_ch,
                            cond_in_channels=2,
                            cond_hidden_ch=out_ch,
                            init_scale=init_scale,
                        )
                    )
                else:
                    modules.append(ResnetBlock(in_ch=in_ch + skip_ch, out_ch=out_ch))

                in_ch = out_ch

            if all_resolutions[i_level] in attn_resolutions:
                modules.append(AttnBlock(channels=in_ch))
            # 交叉注意力引入y_cond
            if self.use_selector and i_level in self.selector_attn_levels:
                modules.append(BottleneckCrossAttention(channels=in_ch, cond_in_channels=2))

            if progressive != 'none':
                if i_level == num_resolutions - 1:
                    if progressive == 'output_skip':
                        modules.append(nn.GroupNorm(num_groups=min(in_ch // 4, 32),
                            num_channels=in_ch, eps=1e-6))
                        modules.append(conv3x3(in_ch, channels, init_scale=init_scale))
                        pyramid_ch = channels
                    elif progressive == 'residual':
                        modules.append(nn.GroupNorm(num_groups=min(in_ch // 4, 32), num_channels=in_ch, eps=1e-6))
                        modules.append(conv3x3(in_ch, in_ch, bias=True))
                        pyramid_ch = in_ch
                    else:
                        raise ValueError(f'{progressive} is not a valid name.')
                else:
                    if progressive == 'output_skip':
                        modules.append(nn.GroupNorm(num_groups=min(in_ch // 4, 32),
                            num_channels=in_ch, eps=1e-6))
                        modules.append(conv3x3(in_ch, channels, bias=True, init_scale=init_scale))
                        pyramid_ch = channels
                    elif progressive == 'residual':
                        modules.append(pyramid_upsample(in_ch=pyramid_ch, out_ch=in_ch))
                        pyramid_ch = in_ch
                    else:
                        raise ValueError(f'{progressive} is not a valid name')

            if i_level != 0:
                if resblock_type == 'ddpm':
                    modules.append(Upsample(in_ch=in_ch))
                else:
                    modules.append(ResnetBlock(in_ch=in_ch, up=True))

        assert not hs_c

        if progressive != 'output_skip':
            modules.append(nn.GroupNorm(num_groups=min(in_ch // 4, 32), num_channels=in_ch, eps=1e-6))
            modules.append(conv3x3(in_ch, channels, init_scale=init_scale))

        self.all_modules = nn.ModuleList(modules)
        

    def forward(self, x, y, t, y_cond=None):
        y_cond = y if y_cond is None else y_cond
        y_hint = self.evidence_selector(y, y_cond) if self.use_selector else y_cond

        # timestep/noise_level embedding; only for continuous training
        modules = self.all_modules
        m_idx = 0

        # Convert real and imaginary parts of (x,y) into four channel dimensions
        x = torch.cat((x.real, x.imag, y.real, y.imag), dim=1)

        if self.embedding_type == 'fourier':
            # Gaussian Fourier features embeddings.
            used_sigmas = t
            temb = modules[m_idx](torch.log(used_sigmas))
            m_idx += 1

        elif self.embedding_type == 'positional':
            # Sinusoidal positional embeddings.
            timesteps = t
            used_sigmas = self.sigmas[t.long()]
            temb = layers.get_timestep_embedding(timesteps, self.nf)

        else:
            raise ValueError(f'embedding type {self.embedding_type} unknown.')

        temb = modules[m_idx](temb)
        m_idx += 1
        temb = modules[m_idx](self.act(temb))
        m_idx += 1

        # Downsampling block
        input_pyramid = None
        if self.progressive_input != 'none':
            input_pyramid = x

        # Input layer: Conv2d: 4ch -> 128ch
        hs = [modules[m_idx](x)]
        m_idx += 1

        # Down path in U-Net 下采样
        for i_level in range(self.num_resolutions):
            # Residual blocks for this resolution
            for i_block in range(self.num_res_blocks):
                h = modules[m_idx](hs[-1], temb)
                m_idx += 1
                # Attention layer (optional)
                if h.shape[-2] in self.attn_resolutions: # edit: check H dim (-2) not W dim (-1)
                    h = modules[m_idx](h)
                    m_idx += 1
                hs.append(h)

            # Downsampling
            if i_level != self.num_resolutions - 1:
                if self.resblock_type == 'ddpm':
                    h = modules[m_idx](hs[-1])
                    m_idx += 1
                else:
                    h = modules[m_idx](hs[-1], temb)
                    m_idx += 1

                if self.progressive_input == 'input_skip':   # Combine h with x
                    input_pyramid = self.pyramid_downsample(input_pyramid)
                    h = modules[m_idx](input_pyramid, h)
                    m_idx += 1

                elif self.progressive_input == 'residual':
                    input_pyramid = modules[m_idx](input_pyramid)
                    m_idx += 1
                    if self.skip_rescale:
                        input_pyramid = (input_pyramid + h) / np.sqrt(2.)
                    else:
                        input_pyramid = input_pyramid + h
                    h = input_pyramid
                hs.append(h)

        h = hs[-1] # actualy equal to: h = h  # bottleneck 输入

        h = modules[m_idx](h, temb)  # ResNet block
        m_idx += 1
        h = modules[m_idx](h)  # Attention block
        m_idx += 1
        if self.cross_attn:
            # NEW: Bottleneck Cross-Attention
            # 使用原始含噪语音 y 的时频信息对 h 做一次 cross-attention
            cond_for_bottleneck = y_hint if self.use_selector else y_cond
            h = self.bottleneck_cross_attn(h, cond_for_bottleneck) #h: (B, C, H, W), cond: (B, 1, H_y, W_y) complex

        h = modules[m_idx](h, temb)  # ResNet block
        m_idx += 1

        pyramid = None

        # Upsampling block
        for i_level in reversed(range(self.num_resolutions)):
            for i_block in range(self.num_res_blocks + 1):
                h_in = torch.cat([h, hs.pop()], dim=1)  #被改动
                
                if self.film_modulate and i_level in self.film_levels:
                    h = modules[m_idx](h_in, temb, y_cond=y_cond)
                else:
                    h = modules[m_idx](h_in, temb)

                m_idx += 1

            # edit: from -1 to -2
            if h.shape[-2] in self.attn_resolutions:
                h = modules[m_idx](h)
                m_idx += 1

            if self.use_selector and i_level in self.selector_attn_levels:
                h = modules[m_idx](h, y_hint)
                m_idx += 1

            # 渐进式的上采样：将上采样过程中的低分辨率特征与高分特征进行结合
            # output_skip：使用跳跃连接的方式进行上采样。上一层的特征会跳跃到当前层，帮助恢复分辨率并保留细节信息。
            # residual：使用残差连接方式进行上采样。当前层的特征和上一层的特征相加。
            if self.progressive != 'none':
                if i_level == self.num_resolutions - 1:
                    if self.progressive == 'output_skip':
                        pyramid = self.act(modules[m_idx](h))  # GroupNorm
                        m_idx += 1
                        pyramid = modules[m_idx](pyramid)  # Conv2D: 256 -> 4
                        m_idx += 1
                    elif self.progressive == 'residual':
                        pyramid = self.act(modules[m_idx](h))
                        m_idx += 1
                        pyramid = modules[m_idx](pyramid)
                        m_idx += 1
                    else:
                        raise ValueError(f'{self.progressive} is not a valid name.')
                else:
                    if self.progressive == 'output_skip':
                        pyramid = self.pyramid_upsample(pyramid)  # Upsample
                        pyramid_h = self.act(modules[m_idx](h))  # GroupNorm
                        m_idx += 1
                        pyramid_h = modules[m_idx](pyramid_h)
                        m_idx += 1
                        pyramid = pyramid + pyramid_h
                    elif self.progressive == 'residual':
                        pyramid = modules[m_idx](pyramid)
                        m_idx += 1
                        if self.skip_rescale:
                            pyramid = (pyramid + h) / np.sqrt(2.)
                        else:
                            pyramid = pyramid + h
                        h = pyramid
                    else:
                        raise ValueError(f'{self.progressive} is not a valid name')

            # Upsampling Layer
            if i_level != 0:
                if self.resblock_type == 'ddpm':
                    h = modules[m_idx](h)
                    m_idx += 1
                else:
                    h = modules[m_idx](h, temb)  # Upspampling
                    m_idx += 1

        assert not hs

        if self.progressive == 'output_skip':
            h = pyramid
        else:
            h = self.act(modules[m_idx](h))
            m_idx += 1
            h = modules[m_idx](h)
            m_idx += 1

        assert m_idx == len(modules), "Implementation error"
        
        h = self.output_layer(h)
        h = torch.permute(h, (0, 2, 3, 1)).contiguous()

        # Convert back to complex number
        h = torch.view_as_complex(h)[:,None, :, :]

        return h
