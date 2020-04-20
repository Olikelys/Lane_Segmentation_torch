#!usr/bin/python
# -*- encoding: utf-8 -*-
"""
@Time           : 2020/4/18 上午10:37
@User           : kang
@Author         : BiKang Peng
@ProjectName    : Lane_Segmentation_torch
@FileName       : decoder.py 
@Software       : PyCharm   
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.gn import GroupNorm
from models.frn import FilterResponseNorm2d

class Decoder(nn.Module):
    def __init__(self, num_classes=8, backbone='resnet', norm_layer=None):
        super(Decoder, self).__init__()
        if backbone is 'resnet' or backbone is 'iresnet':
            low_level_inplanes = 256
        elif backbone is 'resgroup' or backbone is 'iresgroup' or backbone is 'xception':
            low_level_inplanes = 128
        else:
            raise NotImplementedError

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        elif norm_layer is 'gn':
            norm_layer = GroupNorm
        elif norm_layer is 'frn':
            norm_layer = FilterResponseNorm2d

        self.conv1 = nn.Conv2d(low_level_inplanes, 48, 1, bias=False)
        self.bn1 = norm_layer(48)
        self.relu = nn.ReLU(inplace=True)

        self.last_conv = nn.Sequential(
            nn.Conv2d(304, 256, kernel_size=3, stride=1, padding=1, bias=False),
            norm_layer(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),

            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False),
            norm_layer(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),

            nn.Conv2d(256, num_classes, kernel_size=1, stride=1)
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, GroupNorm, FilterResponseNorm2d)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x, low_level_feat):
        low_level_feat = self.conv1(low_level_feat)
        low_level_feat = self.bn1(low_level_feat)
        low_level_feat = self.relu(low_level_feat)

        x = F.interpolate(x, size=low_level_feat.size()[2:], mode='bilinear', align_corners=True)
        x = torch.cat((x, low_level_feat), dim=1)
        x = self.last_conv(x)

        return x


