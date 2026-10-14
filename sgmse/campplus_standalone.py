# Copyright (c) 2023 Hongji Wang; 2024 Zhengyang Chen (original WeSpeaker CAM++).
# Independent in-project adaptation by OpenAI, 2026.
# Licensed under the Apache License, Version 2.0.
# Original implementation: https://github.com/wenet-e2e/wespeaker/blob/master/wespeaker/models/campplus.py
# CAM++ paper: Wang et al., "CAM++: A Fast and Efficient Network for Speaker Verification Using Context-Aware Masking".
"""Standalone CAM++ architecture, preserving WeSpeaker model state_dict keys.

No dependency on ``wespeaker`` or ``s3prl``.  Supports the official
Wespeaker/wespeaker-voxceleb-campplus model config: 80-bin FBank, 512-D
embedding, and TSTP pooling.  Based on the Apache-2.0 WeSpeaker design.
"""
from collections import OrderedDict

import torch
from torch import nn
from torch.nn import functional as F


def get_nonlinear(config_str, channels):
    layer = nn.Sequential()
    for name in config_str.split('-'):
        if name == 'relu':
            layer.add_module('relu', nn.ReLU(inplace=True))
        elif name == 'prelu':
            layer.add_module('prelu', nn.PReLU(channels))
        elif name == 'batchnorm':
            layer.add_module('batchnorm', nn.BatchNorm1d(channels))
        elif name == 'batchnorm_':
            layer.add_module('batchnorm', nn.BatchNorm1d(channels, affine=False))
        else:
            raise ValueError(f'Unsupported activation: {name}')
    return layer


class TSTP(nn.Module):
    """WeSpeaker-style temporal statistics pooling."""
    def __init__(self, in_dim=0, **kwargs):
        super().__init__()
        self.in_dim = in_dim

    def forward(self, x):
        mean = x.mean(dim=-1).flatten(start_dim=1)
        std = torch.sqrt(torch.var(x, dim=-1) + 1e-7).flatten(start_dim=1)
        return torch.cat([mean, std], dim=1)

    def get_out_dim(self):
        return self.in_dim * 2


class TDNNLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, bias=False, config_str='batchnorm-relu'):
        super().__init__()
        if padding < 0:
            assert kernel_size % 2 == 1
            padding = (kernel_size - 1) // 2 * dilation
        self.linear = nn.Conv1d(in_channels, out_channels, kernel_size,
                                stride=stride, padding=padding, dilation=dilation, bias=bias)
        self.nonlinear = get_nonlinear(config_str, out_channels)

    def forward(self, x):
        return self.nonlinear(self.linear(x))


class CAMLayer(nn.Module):
    def __init__(self, bn_channels, out_channels, kernel_size, stride, padding,
                 dilation, bias, reduction=2):
        super().__init__()
        self.linear_local = nn.Conv1d(bn_channels, out_channels, kernel_size,
                                      stride=stride, padding=padding,
                                      dilation=dilation, bias=bias)
        self.linear1 = nn.Conv1d(bn_channels, bn_channels // reduction, 1)
        self.relu = nn.ReLU(inplace=True)
        self.linear2 = nn.Conv1d(bn_channels // reduction, out_channels, 1)
        self.sigmoid = nn.Sigmoid()

    def seg_pooling(self, x, seg_len: int = 100, stype: str = 'avg'):
        if stype == 'avg':
            seg = F.avg_pool1d(x, kernel_size=seg_len, stride=seg_len, ceil_mode=True)
        elif stype == 'max':
            seg = F.max_pool1d(x, kernel_size=seg_len, stride=seg_len, ceil_mode=True)
        else:
            raise ValueError(stype)
        shape = seg.shape
        seg = seg.unsqueeze(-1).expand(shape[0], shape[1], shape[2], seg_len)
        seg = seg.reshape(shape[0], shape[1], -1)
        return seg[..., :x.shape[-1]]

    def forward(self, x):
        y = self.linear_local(x)
        context = x.mean(-1, keepdim=True) + self.seg_pooling(x)
        context = self.relu(self.linear1(context))
        mask = self.sigmoid(self.linear2(context))
        return y * mask


class CAMDenseTDNNLayer(nn.Module):
    def __init__(self, in_channels, out_channels, bn_channels, kernel_size,
                 stride=1, dilation=1, bias=False, config_str='batchnorm-relu'):
        super().__init__()
        assert kernel_size % 2 == 1
        padding = (kernel_size - 1) // 2 * dilation
        self.nonlinear1 = get_nonlinear(config_str, in_channels)
        self.linear1 = nn.Conv1d(in_channels, bn_channels, 1, bias=False)
        self.nonlinear2 = get_nonlinear(config_str, bn_channels)
        self.cam_layer = CAMLayer(bn_channels, out_channels, kernel_size,
                                  stride=stride, padding=padding, dilation=dilation, bias=bias)

    def bn_function(self, x):
        return self.linear1(self.nonlinear1(x))

    def forward(self, x):
        return self.cam_layer(self.nonlinear2(self.bn_function(x)))


class CAMDenseTDNNBlock(nn.ModuleList):
    def __init__(self, num_layers, in_channels, out_channels, bn_channels,
                 kernel_size, stride=1, dilation=1, bias=False,
                 config_str='batchnorm-relu'):
        super().__init__()
        for i in range(num_layers):
            self.add_module(f'tdnnd{i + 1}', CAMDenseTDNNLayer(
                in_channels + i * out_channels, out_channels, bn_channels,
                kernel_size, stride=stride, dilation=dilation, bias=bias,
                config_str=config_str))

    def forward(self, x):
        for layer in self:
            x = torch.cat([x, layer(x)], dim=1)
        return x


class TransitLayer(nn.Module):
    def __init__(self, in_channels, out_channels, bias=True,
                 config_str='batchnorm-relu'):
        super().__init__()
        self.nonlinear = get_nonlinear(config_str, in_channels)
        self.linear = nn.Conv1d(in_channels, out_channels, 1, bias=bias)

    def forward(self, x):
        return self.linear(self.nonlinear(x))


class DenseLayer(nn.Module):
    def __init__(self, in_channels, out_channels, bias=False,
                 config_str='batchnorm-relu'):
        super().__init__()
        self.linear = nn.Conv1d(in_channels, out_channels, 1, bias=bias)
        self.nonlinear = get_nonlinear(config_str, out_channels)

    def forward(self, x):
        if x.ndim == 2:
            return self.nonlinear(self.linear(x.unsqueeze(-1)).squeeze(-1))
        return self.nonlinear(self.linear(x))


class BasicResBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3,
                               stride=(stride, 1), padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes * self.expansion, kernel_size=1,
                          stride=(stride, 1), bias=False),
                nn.BatchNorm2d(planes * self.expansion),
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = out + self.shortcut(x)
        return F.relu(out)


class FCM(nn.Module):
    def __init__(self, block, num_blocks, m_channels=32, feat_dim=80):
        super().__init__()
        self.in_planes = m_channels
        self.conv1 = nn.Conv2d(1, m_channels, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(m_channels)
        self.layer1 = self._make_layer(block, m_channels, num_blocks[0], stride=2)
        self.layer2 = self._make_layer(block, m_channels, num_blocks[1], stride=2)
        self.conv2 = nn.Conv2d(m_channels, m_channels, kernel_size=3,
                               stride=(2, 1), padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(m_channels)
        self.out_channels = m_channels * (feat_dim // 8)

    def _make_layer(self, block, planes, num_blocks, stride):
        layers = []
        for s in [stride] + [1] * (num_blocks - 1):
            layers.append(block(self.in_planes, planes, s))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x.unsqueeze(1))))
        out = self.layer1(out)
        out = self.layer2(out)
        out = F.relu(self.bn2(self.conv2(out)))
        sh = out.shape
        return out.reshape(sh[0], sh[1] * sh[2], sh[3])


class CAMPPlus(nn.Module):
    """Official WeSpeaker CAM++ topology with 80-dim fbank + 512-dim embed."""
    def __init__(self, feat_dim=80, embed_dim=512, pooling_func='TSTP',
                 growth_rate=32, bn_size=4, init_channels=128,
                 config_str='batchnorm-relu'):
        super().__init__()
        if pooling_func != 'TSTP':
            raise ValueError('Standalone checkpoint currently supports TSTP only')
        self.head = FCM(BasicResBlock, [2, 2], feat_dim=feat_dim)
        channels = self.head.out_channels
        self.xvector = nn.Sequential(OrderedDict([
            ('tdnn', TDNNLayer(channels, init_channels, 5, stride=2,
                               dilation=1, padding=-1, config_str=config_str))
        ]))
        channels = init_channels
        for i, (num_layers, kernel_size, dilation) in enumerate(
                zip((12, 24, 16), (3, 3, 3), (1, 2, 2))):
            block = CAMDenseTDNNBlock(num_layers, channels, growth_rate,
                                      bn_size * growth_rate, kernel_size,
                                      dilation=dilation, config_str=config_str)
            self.xvector.add_module(f'block{i + 1}', block)
            channels += num_layers * growth_rate
            self.xvector.add_module(f'transit{i + 1}', TransitLayer(
                channels, channels // 2, bias=False, config_str=config_str))
            channels //= 2
        self.xvector.add_module('out_nonlinear', get_nonlinear(config_str, channels))
        self.pool = TSTP(in_dim=channels)
        self.pool_out_dim = self.pool.get_out_dim()
        self.xvector.add_module('stats', self.pool)
        self.xvector.add_module('dense', DenseLayer(self.pool_out_dim, embed_dim,
                                                    config_str='batchnorm_'))
        for m in self.modules():
            if isinstance(m, (nn.Conv1d, nn.Linear)):
                nn.init.kaiming_normal_(m.weight.data)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        return self.xvector(self.head(x.permute(0, 2, 1)))
