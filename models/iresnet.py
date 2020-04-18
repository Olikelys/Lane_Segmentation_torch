#!usr/bin/python
# -*- encoding: utf-8 -*-
"""
@Time           : 2020/4/17 下午1:55
@User           : kang
@Author         : BiKang Peng
@ProjectName    : Lane_Segmentation_torch
@FileName       : iresnet.py 
@Software       : PyCharm   
"""

import torch
import torch.nn as nn
import os
import sys

sys.path.append('../')
from bn.gn import GroupNorm
from bn.frn import FilterResponseNorm2d


def conv3x3(in_planes, out_planes, stride=1):
    """
    自定义3x3卷积
    @param in_planes: 输入通道
    @param out_planes: 输出通道
    @param stride: 步长
    @return:
    """
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


def conv1x1(in_planes, out_planes, stride=1):
    """
    自定义1x1卷积
    @param in_planes: 输入通道
    @param out_planes: 输出通道
    @param stride: 步长
    @return:
    """
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, norm_layer='rfn', start_block=False,
                 end_block=False, exclude_bn0=False):
        """
        定义基本残插块
        优化了resnet的残插模块， 分别为:
        Start ResBlock: (3x3 conv + BN + ReLU + 3x3 conv + BN ) + x
        Middle ResBlock: (BN + ReLU + 3x3 conv + BN + ReLU + 3x3 conv ) + x
        End ResBlock: ((BN + ReLU + 3x3 conv + BN + ReLU + 3x3 conv ) + x) + BN + ReLU
        @param inplanes:
        @param planes:
        @param stride:
        @param downsample:
        @param norm_layer:
        @param start_block:
        @param end_block:
        @param exclude_bn0:
        """
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        elif norm_layer is 'gn':
            norm_layer = GroupNorm
        elif norm_layer is 'frn':
            norm_layer = FilterResponseNorm2d

        if not start_block and not exclude_bn0:
            self.bn0 = norm_layer(inplanes)
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)

        if start_block:
            self.bn2 = norm_layer(planes)
        if end_block:
            self.bn2 = norm_layer(planes)

        self.downsample = downsample
        self.stride = stride

        self.start_block = start_block
        self.end_block = end_block
        self.exclude_bn0 = exclude_bn0

    def forward(self, x):
        identify = x

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

        if self.start_block:
            out = self.bn2(out)

        if self.downsample:
            identify = self.downsample(x)

        out += identify

        if self.end_block:
            out = self.bn2(out)
            out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, norm_layer=None, start_block=False,
                 end_block=False, exclude_bn0=False):
        """
        优化了resnet的残插模块， 分别为:
        Start ResBlock: (1x1 conv + BN + ReLU + 3x3 conv + BN + ReLU + 1x1 conv + BN) + x
        Middle ResBlock: (BN + ReLU + 1x1 conv + BN + ReLU + 3x3 conv + BN + ReLU + 1x1 conv) + x
        End ResBlock: ((BN + ReLU + 1x1 conv + BN + ReLU + 3x3 conv + BN + ReLU + 1x1 conv) + x) + BN + ReLU
        @param inplanes:
        @param planes:
        @param stride:
        @param downsample:
        @param norm_layer:
        @param start_block:
        @param end_block:
        @param exclude_bn0:
        """
        super(Bottleneck, self).__init__()
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
        self.conv2 = conv3x3(planes, planes, stride)
        self.bn2 = norm_layer(planes)
        self.conv3 = conv1x1(planes, planes * self.expansion)

        if start_block:
            self.bn3 = norm_layer(planes * self.expansion)

        if end_block:
            self.bn3 = norm_layer(planes * self.expansion)

        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

        self.start_block = start_block
        self.end_block = end_block
        self.exclude_bn0 = exclude_bn0

    def forward(self, x):
        identify = x

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

        if self.downsample:
            identify = self.downsample(x)

        out += identify

        if self.end_block:
            out = self.bn3(out)
            out = self.relu(out)

        return out


class iResNet(nn.Module):
    def __init__(self, block, layers, norm_layer=None):
        """
        构建iResNet网络
        @param block:
        @param layers:
        @param norm_layer:
        """
        super(iResNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        elif norm_layer is 'gn':
            norm_layer = GroupNorm
        elif norm_layer is 'frn':
            norm_layer = FilterResponseNorm2d

        self.inplanes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = norm_layer(64)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(block, 64, layers[0], stride=2, norm_layer=norm_layer)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, norm_layer=norm_layer)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2, norm_layer=norm_layer)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2, norm_layer=norm_layer)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, GroupNorm, FilterResponseNorm2d)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1, norm_layer=None):
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        elif norm_layer is 'gn':
            norm_layer = GroupNorm
        elif norm_layer is 'frn':
            norm_layer = FilterResponseNorm2d

        downsample = None
        if stride != 1 and self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.MaxPool2d(kernel_size=3, stride=stride, padding=1),
                conv1x1(self.inplanes, planes * block.expansion),
                norm_layer(planes * block.expansion)
            )
        elif self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion),
                norm_layer(planes * block.expansion)
            )
        elif stride != 1:
            downsample = nn.MaxPool2d(kernel_size=3, stride=stride, padding=1)

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, norm_layer, start_block=True))
        self.inplanes = planes * block.expansion
        exclude_bn0 = True
        for _ in range(1, blocks - 1):
            layers.append(block(self.inplanes, planes, norm_layer=norm_layer, exclude_bn0=exclude_bn0))
            exclude_bn0 = False
        layers.append(block(self.inplanes, planes, norm_layer=norm_layer, end_block=True, exclude_bn0=exclude_bn0))

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


def iresnet18(norm_layer=None, **kwargs):
    model = iResNet(BasicBlock, [2, 2, 2, 2], norm_layer=norm_layer, **kwargs)
    return model


def iresnet34(norm_layer=None, **kwargs):
    model = iResNet(BasicBlock, [3, 4, 6, 3], norm_layer=norm_layer, **kwargs)
    return model


def iresnet50(norm_layer=None, **kwargs):
    model = iResNet(Bottleneck, [3, 4, 6, 3], norm_layer=norm_layer, **kwargs)
    return model


def iresnet101(norm_layer=None, **kwargs):
    model = iResNet(Bottleneck, [3, 4, 23, 3], norm_layer=norm_layer, **kwargs)
    return model


def iresnet152(norm_layer=None, **kwargs):
    model = iResNet(Bottleneck, [3, 8, 36, 3], norm_layer=norm_layer, **kwargs)
    return model


def iresnet200(norm_layer=None, **kwargs):
    model = iResNet(Bottleneck, [3, 24, 36, 3], norm_layer=norm_layer, **kwargs)
    return model


def iresnet302(norm_layer=None, **kwargs):
    model = iResNet(Bottleneck, [4, 34, 58, 4], norm_layer=norm_layer, **kwargs)
    return model


def iresnet404(norm_layer=None, **kwargs):
    model = iResNet(Bottleneck, [4, 46, 80, 4], norm_layer=norm_layer, **kwargs)
    return model


def iresnet1001(norm_layer=None, **kwargs):
    model = iResNet(Bottleneck, [4, 155, 170, 4], norm_layer=norm_layer, **kwargs)
    return model


if __name__ == '__main__':
    from torchsummary import summary

    model = iresnet101(norm_layer='frn')
    summary(model.cuda(), (3, 256, 256))
