#!usr/bin/python
# -*- encoding: utf-8 -*-
"""
@Time           : 2020/4/18 上午9:02
@User           : kang
@Author         : BiKang Peng
@ProjectName    : Lane_Segmentation_torch
@FileName       : aspp.py 
@Software       : PyCharm   
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.gn import GroupNorm
from models.frn import FilterResponseNorm2d


class _ASPPModule(nn.Module):
    def __init__(self, inplanes, planes, kernel_size, padding, dilation, norm_layer):
        """
        定义ASPP网络中子网络模块
        @param inplanes:
        @param planes:
        @param kernel_size:
        @param padding:
        @param dilation:
        @param norm_layer:
        """
        super(_ASPPModule, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        elif norm_layer is 'gn':
            norm_layer = GroupNorm
        elif norm_layer is 'frn':
            norm_layer = FilterResponseNorm2d

        self.atrous_conv = nn.Conv2d(
            in_channels=inplanes,
            out_channels=planes,
            kernel_size=kernel_size,
            stride=1,
            padding=padding,
            dilation=dilation,
            bias=False
        )
        self.bn = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, GroupNorm, FilterResponseNorm2d)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.atrous_conv(x)
        x = self.bn(x)
        x = self.relu(x)

        return x


class ASPP(nn.Module):
    def __init__(self, backbone='resnet-50', output_stride=16, norm_layer=None):
        """
        构建ASPP网络
        @param backbone:
        @param output_stride:
        @param norm_layer:
        """
        super(ASPP, self).__init__()
        if backbone is 'resnet' or backbone is 'iresnet' or backbone is 'xception':
            inplanes = 2048
        elif backbone is 'resgroup' or backbone is 'iresgroup':
            inplanes = 1024

        if output_stride == 16:
            dilation = [1, 6, 12, 18]
        elif output_stride == 8:
            dilation = [1, 12, 24, 36]
        else:
            raise NotImplementedError

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        elif norm_layer is 'gn':
            norm_layer = GroupNorm
        elif norm_layer is 'frn':
            norm_layer = FilterResponseNorm2d

        self.aspp1 = _ASPPModule(inplanes, 256, 1, padding=0, dilation=dilation[0], norm_layer=norm_layer)
        self.aspp2 = _ASPPModule(inplanes, 256, 3, padding=dilation[1], dilation=dilation[1], norm_layer=norm_layer)
        self.aspp3 = _ASPPModule(inplanes, 256, 3, padding=dilation[2], dilation=dilation[2], norm_layer=norm_layer)
        self.aspp4 = _ASPPModule(inplanes, 256, 3, padding=dilation[3], dilation=dilation[3], norm_layer=norm_layer)
        self.global_avg_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(inplanes, 256, 1, stride=1, bias=False),
            norm_layer(256),
            nn.ReLU(inplace=True)
        )

        self.conv1 = nn.Conv2d(1280, 256, 1, bias=False)
        self.bn1 = norm_layer(256)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(0.5)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, GroupNorm, FilterResponseNorm2d)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x1 = self.aspp1(x)
        x2 = self.aspp2(x)
        x3 = self.aspp3(x)
        x4 = self.aspp4(x)
        x5 = self.global_avg_pool(x)
        x5 = F.interpolate(x5, size=x4.size()[2:], mode='bilinear', align_corners=True)
        x = torch.cat((x1, x2, x3, x4, x5), dim=1)

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.dropout(x)
        return x

def aspp(backbone, output_stride, norm_layer=None):
    model = ASPP(backbone=backbone, output_stride=output_stride, norm_layer=norm_layer)
    return model





