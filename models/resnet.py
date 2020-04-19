#!usr/bin/python
# -*- encoding: utf-8 -*-
"""
@Time           : 2020/4/17 下午2:37
@User           : kang
@Author         : BiKang Peng
@ProjectName    : Lane_Segmentation_torch
@FileName       : resnet.py 
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
    定义3x3卷积层
    @param in_planes: 输入通道
    @param out_planes: 输出通道
    @param stride: 步长
    @return:
    """
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


def conv1x1(in_planes, out_planes, stride=1):
    """
    定义1x1卷积
    @param in_planes: 输入通道
    @param out_planes: 输出通道
    @param stride: 步长
    @return:
    """
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, norm_layer=None):
        """
        定义一个基本残插块 3x3 conv + 3x3 conv
        @param inplanes:
        @param planes:
        @param stride:
        @param downsample:
        @param norm_layer:
        """
        super(BasicBlock, self).__init__()
        if norm_layer is 'nn':
            norm_layer = nn.BatchNorm2d
        elif norm_layer is 'gn':
            norm_layer = GroupNorm
        elif norm_layer is 'frn':
            norm_layer = FilterResponseNorm2d

        self.use_bn = norm_layer is not None
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identify = x

        out = self.conv1(x)
        if self.use_bn:
            out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        if self.use_bn:
            out = self.bn2(out)

        if self.downsample is not None:
            identify = self.downsample(x)

        out += identify
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, norm_layer=None):
        """
        定义残插模块 1x1 + 3x3 + 1x1 conv
        @param inplanes:
        @param planes:
        @param stride:
        @param downsample:
        @param norm_layer:
        """
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        elif norm_layer is 'gn':
            norm_layer = GroupNorm
        elif norm_layer is 'frn':
            norm_layer = FilterResponseNorm2d

        self.conv1 = conv1x1(inplanes, planes)
        self.bn1 = norm_layer(planes)
        self.conv2 = conv3x3(planes, planes, stride)
        self.bn2 = norm_layer(planes)
        self.conv3 = conv1x1(planes, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identify = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identify = self.downsample(x)

        out += identify
        out = self.relu(out)

        return out


class ResNet(nn.Module):
    def __init__(self, block, layers, zero_init_residual=True, norm_layer=None):
        """
        构建ResNet
        @param block:
        @param layers:
        @param norm_layer:
        """
        super(ResNet, self).__init__()
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
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0], norm_layer=norm_layer)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, norm_layer=norm_layer)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2, norm_layer=norm_layer)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2, norm_layer=norm_layer)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, GroupNorm, FilterResponseNorm2d)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, norm_layer=None):
        """
        构建网络层
        @param block: 基本模块
        @param planes: 输入通道
        @param blocks: n个循环块
        @param stride:
        @param norm_layer:
        @return:
        """
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        elif norm_layer is 'gn':
            norm_layer = GroupNorm
        elif norm_layer is 'frn':
            norm_layer = FilterResponseNorm2d
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion)
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        low_level_feat = x

        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        return x, low_level_feat


def resnet18(norm_layer=None, **kwargs):
    model = ResNet(BasicBlock, [2, 2, 2, 2], norm_layer=norm_layer, **kwargs)
    return model


def resnet34(norm_layer=None, **kwargs):
    model = ResNet(BasicBlock, [3, 4, 6, 3], norm_layer=norm_layer, **kwargs)
    return model


def resnet50(norm_layer=None, **kwargs):
    model = ResNet(Bottleneck, [3, 4, 6, 3], norm_layer=norm_layer, **kwargs)
    return model


def resnet101(norm_layer=None, **kwargs):
    model = ResNet(Bottleneck, [3, 4, 23, 3], norm_layer=norm_layer, **kwargs)
    return model


def resnet152(norm_layer=None, **kwargs):
    model = ResNet(Bottleneck, [3, 8, 36, 3], norm_layer=norm_layer, **kwargs)
    return model


def resnet200(norm_layer=None, **kwargs):
    model = ResNet(Bottleneck, [3, 24, 36, 3], norm_layer=norm_layer, **kwargs)
    return model


if __name__ == '__main__':
    from torchsummary import summary

    model = resnet101(norm_layer='frn')
    summary(model.cuda(), (3, 256, 256))
