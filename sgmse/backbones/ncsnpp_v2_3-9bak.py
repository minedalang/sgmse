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

class BottleneckCrossAttention(nn.Module):
    """
    在 U-Net 瓶颈处，用原始含噪语音 y 对主干特征 h 做一次 cross-attention：
      - h: (B, C, H, W)  主干特征（瓶颈）
      - y: (B, 1, H_y, W_y)  复数谱（complex tensor）
    - 先把 y -> cond(y) （非线性卷积），作为 Q 的输入；
    - h 归一化后生成 K/V；
    - Attention: Q(cond(y)) 在 K/V(h) 上查找“在 h 的表征下，y 的哪些位置重要”，
    再把得到的表示映射回 h 的空间上，并 residual 加到 h 上。
    """
    def __init__(self, channels: int, cond_in_channels: int = 2):
        super().__init__()
        self.channels = channels

        # 对 h 做一个简单的归一化，稳定注意力（模仿 AttnBlockpp 里的 GroupNorm）
        self.norm = nn.GroupNorm(
            num_groups=min(channels // 4, 32),
            num_channels=channels,
            eps=1e-6
        )

        # 对 y 的时频特征做一个非线性变换: conv + ReLU
        self.cond_net = nn.Sequential(
            nn.Conv2d(cond_in_channels, channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

        
        self.q_proj = nn.Conv2d(channels, channels, kernel_size=1)
        self.k_proj = nn.Conv2d(channels, channels, kernel_size=1)
        self.v_proj = nn.Conv2d(channels, channels, kernel_size=1)
        self.proj_out = nn.Conv2d(channels, channels, kernel_size=1)

        # 初始化：跟 NCSNpp 其他卷积保持一致
        for m in [self.cond_net[0], self.q_proj, self.k_proj, self.v_proj, self.proj_out]:
            m.weight.data = default_initializer()(m.weight.shape)
            nn.init.zeros_(m.bias)

    def forward(self, h: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        h: (B, C, H, W)
        y: (B, 1, H_y, W_y) complex tensor
        """
        B, C, H, W = h.shape

        # 1) 把 y 拆成 real/imag，拼成条件特征
        cond = torch.cat([y.real, y.imag], dim=1)  # (B, 2, H_y, W_y)

        # 2) 把y 的时频分辨率与bottleneck特征对齐
        if cond.shape[2] != H or cond.shape[3] != W:
            cond = F.interpolate(cond, size=(H, W), mode="bilinear", align_corners=False)
        
        # 3) 对 y 的时频特征做非线性变换
        cond = self.cond_net(cond)  # (B, C, H, W)

        # 4) 生成 Q / K / V
        #    Q 从归一化后的 h 出来，K/V 从变换后的 y 出来
        h_norm = self.norm(h)   
        
        q = self.q_proj(cond)       
        k = self.k_proj(h_norm)     
        v = self.v_proj(h_norm)   

        # 5) 展平到 (B, N, C)，N = H * W
        q = q.view(B, C, H * W).transpose(1, 2)  
        k = k.view(B, C, H * W).transpose(1, 2)  
        v = v.view(B, C, H * W).transpose(1, 2) 

        # 6) softmax(QK^T / sqrt(C)) V
        scale = 1.0 / np.sqrt(C)
        attn = torch.softmax(torch.bmm(q, k.transpose(1, 2)) * scale, dim=-1)  # (B, N_q, N_k)
        out = torch.bmm(attn, v)  # (B, N_q, C)

        # 7) reshape 回 2D，再做一个输出投影 + 残差
        out = out.transpose(1, 2).view(B, C, H, W)  # (B, C, H, W)
        out = self.proj_out(out)                    # (B, C, H, W)

        return h + out  # residual connection
    
class TimeFreqFiLM(nn.Module):
    """
    时频 FiLM：通过原始含噪语音 y 对上采样特征 h 进行条件化缩放与偏移。
    """
    def __init__(self, cond_in_channels: int, feature_channels: int):
        super().__init__()

        # 对 y 的时频信息做非线性变换
        self.cond_net = nn.Sequential(
            nn.Conv2d(cond_in_channels, feature_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(feature_channels, 2 * feature_channels, kernel_size=1)
        )

        # 初始化：跟 NCSNpp 其他卷积保持一致
        for m in [self.cond_net[0], self.cond_net[2]]:
            m.weight.data = default_initializer()(m.weight.shape)
            nn.init.zeros_(m.bias)

    def forward(self, h: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        h: (B, C, H, W) 主干特征（上采样的结果）
        y: (B, 1, H_y, W_y) 含噪语音的复数谱（complex tensor）
        """
        B, C, H, W = h.shape

        # 1) 将 y 拼接成 real 和 imag 两个通道
        cond = torch.cat([y.real, y.imag], dim=1)  # (B, 2, H_y, W_y)

        # 2) 如果 y 的时频分辨率和 h 不一致，进行上采样
        if cond.shape[2] != H or cond.shape[3] != W:
            cond = F.interpolate(cond, size=(H, W), mode="bilinear", align_corners=False)

        # 3) 对 y 的时频特征做非线性变换 生成gamma/beta
        cond = self.cond_net(cond)  # (B, 2 * C, H, W)

        # 4) 将变换后的特征分为 gamma 和 beta
        gamma, beta = torch.chunk(cond, 2, dim=1)  # (B, C, H, W) + (B, C, H, W)

        # 5) FiLM 操作：缩放与平移
        h = h * (1.0 + gamma) + beta
        return h
    
@BackboneRegistry.register("ncsnpp_v2")
class NCSNpp_v2(nn.Module):
    """NCSN++ model, adapted from https://github.com/yang-song/score_sde repository"""

    @staticmethod
    def add_argparse_args(parser):
        parser.add_argument("--nf", type=int, default=128)
        parser.add_argument("--ch_mult",type=int, nargs='+', default=[1,1,2,2,2,2,2])
        parser.add_argument("--num_res_blocks", type=int, default=2)
        parser.add_argument("--attn_resolutions", type=int, nargs='+', default=[16])
        parser.add_argument("--cross_attn", action='store_true', default=False)
        parser.add_argument("--film_modulate", action='store_true', default=False)
        return parser
    def __init__(self,
        nf = 128,
        ch_mult = (1, 1, 2, 2, 2, 2, 2),
        num_res_blocks = 2,
        attn_resolutions = (16,),
        cross_attn = False,
        film_modulate=False,
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
            # NEW: 瓶颈处的 cross-attention，用原始含噪语音 y 作为条件
            self.bottleneck_cross_attn = BottleneckCrossAttention(
                channels=in_ch,
                cond_in_channels=2  # y.real + y.imag
            )

        modules.append(ResnetBlock(in_ch=in_ch))
        modules.append(AttnBlock(channels=in_ch))
        modules.append(ResnetBlock(in_ch=in_ch))

        pyramid_ch = 0
        # Upsampling block
        for i_level in reversed(range(num_resolutions)):
            for i_block in range(num_res_blocks + 1):  # +1 blocks in upsampling because of skip connection from combiner (after downsampling)
                out_ch = nf * ch_mult[i_level]
                modules.append(ResnetBlock(in_ch=in_ch + hs_c.pop(), out_ch=out_ch))
                in_ch = out_ch

            if all_resolutions[i_level] in attn_resolutions:
                modules.append(AttnBlock(channels=in_ch))

            # NEW: 为上采样过程添加 FiLM 模块
            if self.film_modulate:
                modules.append(TimeFreqFiLM(
                    cond_in_channels=2,  # y.real + y.imag
                    feature_channels=in_ch   
                ))

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
            h = self.bottleneck_cross_attn(h, y_cond) #h: (B, C, H, W), y: (B, 1, H_y, W_y) complex

        h = modules[m_idx](h, temb)  # ResNet block
        m_idx += 1

        pyramid = None

        # Upsampling block
        for i_level in reversed(range(self.num_resolutions)):
            for i_block in range(self.num_res_blocks + 1):
                h = modules[m_idx](torch.cat([h, hs.pop()], dim=1), temb)
                m_idx += 1

            # edit: from -1 to -2
            if h.shape[-2] in self.attn_resolutions:
                h = modules[m_idx](h)
                m_idx += 1

            # FiLM模块
            if self.film_modulate:
                h = modules[m_idx](h, y_cond) 
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
