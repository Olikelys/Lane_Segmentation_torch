#!usr/bin/python
# -*- encoding: utf-8 -*-
"""
@Time           : 2020/4/17 下午7:54
@User           : kang
@Author         : BiKang Peng
@ProjectName    : Lane_Segmentation_torch
@FileName       : iresgroup.py 
@Software       : PyCharm   
"""
import torch
import torch.nn as nn
import os
import sys

sys.path.append('../')
from bn.gn import GroupNorm
from bn.frn import FilterResponseNorm2d


def conv3x3(in_planes, out_planes, groups=1, stride=1):
    """
    定义3x3组卷积
    @param in_planes:
    @param out_planes:
    @param groups:
    @param stride:
    @return:
    """
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False, groups=groups)


def conv1x1(in_planes, out_planes, stride=1):
    """
    定义1x1组卷积
    @param in_planes:
    @param out_planes:
    @param stride:
    @return:
    """
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class ResGroupBlock(nn.Module):
    reduction = 2

    def __init__(self, inplanes, planes, groups, stride=1, downsample=None, norm_layer=None, start_block=False,
                 end_block=False, exclude_bn0=False):
        """
        定义残插组模块
        @param inplanes:
        @param planes:
        @param groups:
        @param stride:
        @param downsample:
        @param norm_layer:
        @param start_block:
        @param end_block:
        @param exclude_bn0:
        """
        super(ResGroupBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        elif norm_layer is 'gn':
            norm_layer = GroupNorm
        elif norm_layer is 'frn':
            norm_layer = FilterResponseNorm2d

        if not start_block and not exclude_bn0:
            self.bn0 = norm_layer(inplanes)

        self.conv1 = conv1x1(inplanes, planes)
        self.bn1 = norm_layer(planes)
        self.conv2 = conv3x3(planes, planes, groups=groups, stride=stride)
        self.bn2 = norm_layer(planes)
        self.conv3 = conv1x1(planes, planes // self.reduction)

        if start_block:
            self.bn3 = norm_layer(planes // self.reduction)

        if end_block:
            self.bn3 = norm_layer(planes // self.reduction)

        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

        self.start_block = start_block
        self.end_block = end_block
        self.exclude_bn0 = exclude_bn0

    def forward(self, x):
        identity = x

        if self.start_block:
            out = self.conv1(x)
        elif self.exclude_bn0:
            out = self.relu(x)
            out = self.conv1(out)
        else:
            out = self.bn0(x)
            out = self.relu(out)
            out = self.conv1(out)

        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)

        if self.start_block:
            out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity

        if self.end_block:
            out = self.bn3(out)
            out = self.relu(out)

        return out


class iResGroup(nn.Module):

    def __init__(self, block, layers, norm_layer=None, groups=None):
        """
        构建iResGroup网络
        @param block:
        @param layers:
        @param norm_layer:
        @param groups:
        """
        super(iResGroup, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        elif norm_layer is 'gn':
            norm_layer = GroupNorm
        elif norm_layer is 'frn':
            norm_layer = FilterResponseNorm2d
        if groups is None:
            groups = 64

        self.inplanes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = norm_layer(64)
        self.relu = nn.ReLU(inplace=True)

        self.layer1 = self._make_layer(block, 256, layers[0], groups=max(1, groups // 8), stride=2,
                                       norm_layer=norm_layer)
        self.layer2 = self._make_layer(block, 512, layers[1], groups=max(1, groups // 4), stride=2,
                                       norm_layer=norm_layer)
        self.layer3 = self._make_layer(block, 1024, layers[2], groups=max(1, groups // 2), stride=2,
                                       norm_layer=norm_layer)
        self.layer4 = self._make_layer(block, 2048, layers[3], groups=groups, stride=2, norm_layer=norm_layer)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm, FilterResponseNorm2d)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, groups, stride=1, norm_layer=None):
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        elif norm_layer is 'gn':
            norm_layer = GroupNorm
        elif norm_layer is 'frn':
            norm_layer = FilterResponseNorm2d
        downsample = None
        if stride != 1 and self.inplanes != planes // block.reduction:
            downsample = nn.Sequential(
                nn.MaxPool2d(kernel_size=3, stride=stride, padding=1),
                conv1x1(self.inplanes, planes // block.reduction),
                norm_layer(planes // block.reduction),
            )
        elif self.inplanes != planes // block.reduction:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes // block.reduction),
                norm_layer(planes // block.reduction),
            )
        elif stride != 1:
            downsample = nn.MaxPool2d(kernel_size=3, stride=stride, padding=1)

        layers = []
        layers.append(block(self.inplanes, planes, groups, stride, downsample, norm_layer, start_block=True))
        self.inplanes = planes // block.reduction
        exclude_bn0 = True
        for _ in range(1, (blocks - 1)):
            layers.append(block(self.inplanes, planes, groups, norm_layer=norm_layer,
                                exclude_bn0=exclude_bn0))
            exclude_bn0 = False

        layers.append(
            block(self.inplanes, planes, groups, norm_layer=norm_layer, end_block=True, exclude_bn0=exclude_bn0))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.layer1(x)
        low_level_feat = x

        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        return x, low_level_feat


def iresgroup50(norm_layer=None, **kwargs):
    model = iResGroup(ResGroupBlock, [3, 4, 6, 3], norm_layer=norm_layer, **kwargs)
    return model


def iresgroup101(norm_layer=None, **kwargs):
    model = iResGroup(ResGroupBlock, [3, 4, 23, 3], norm_layer=norm_layer, **kwargs)
    return model


def iresgroup152(norm_layer=None, **kwargs):
    model = iResGroup(ResGroupBlock, [3, 8, 36, 3], norm_layer=norm_layer, **kwargs)
    return model


if __name__ == '__main__':
    from torchsummary import summary

    model = iresgroup101(norm_layer='frn')
    summary(model.cuda(), (3, 256, 256))
