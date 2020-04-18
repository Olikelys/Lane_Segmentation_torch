#!usr/bin/python
# -*- encoding: utf-8 -*-
"""
@Time           : 2020/4/17 下午8:33
@User           : kang
@Author         : BiKang Peng
@ProjectName    : Lane_Segmentation_torch
@FileName       : xception.py 
@Software       : PyCharm   
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import sys

sys.path.append('../')
from bn.gn import GroupNorm
from bn.frn import FilterResponseNorm2d


def fixed_padding(inputs, kernel_size, dilation):
    kernel_size_effective = kernel_size + (kernel_size - 1) * (dilation - 1)
    pad_total = kernel_size_effective - 1
    pad_beg = pad_total // 2
    pad_end = pad_total - pad_beg
    padded_inputs = F.pad(inputs, (pad_beg, pad_end, pad_beg, pad_end))
    return padded_inputs


class SeparableConv2d(nn.Module):
    def __init__(self, inplanes, planes, kernel_size=3, stride=1, dilation=1, bias=False, norm_layer=None):
        super(SeparableConv2d, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        elif norm_layer is 'gn':
            norm_layer = GroupNorm
        elif norm_layer is 'frn':
            norm_layer = FilterResponseNorm2d
        self.conv1 = nn.Conv2d(
            in_channels=inplanes,
            out_channels=inplanes,
            kernel_size=kernel_size,
            stride=stride,
            padding=0,
            dilation=dilation,
            groups=inplanes,
            bias=bias
        )
        self.bn = norm_layer(inplanes)
        self.pointwise = nn.Conv2d(
            in_channels=inplanes,
            out_channels=planes,
            kernel_size=1,
            stride=1,
            padding=0,
            dilation=1,
            groups=1,
            bias=bias
        )

    def forward(self, x):
        x = fixed_padding(x, self.conv1.kernel_size[0], dilation=self.conv1.dilation[0])
        x = self.conv1(x)
        x = self.bn(x)
        x = self.pointwise(x)
        return x


class Block(nn.Module):
    def __init__(self, inplanes, planes, reps, stride=1, dilation=1, norm_layer=None, start_with_relu=True,
                 grow_fist=True, is_last=True):
        """
        定义深度可分离卷积块
        @param inplanes:
        @param planes:
        @param reps:
        @param stride:
        @param dilation:
        @param norm_layer:
        @param start_with_relu:
        @param grow_fist:
        @param is_last:
        """
        super(Block, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        elif norm_layer is 'gn':
            norm_layer = GroupNorm
        elif norm_layer is 'frn':
            norm_layer = FilterResponseNorm2d

        if planes != inplanes or stride != 1:
            self.skip = nn.Conv2d(inplanes, planes, kernel_size=1, stride=stride, bias=False)
            self.skipbn = norm_layer(planes)
        else:
            self.skip = None

        self.relu = nn.ReLU(inplace=True)

        rep = []

        filters = inplanes

        if grow_fist:
            rep.append(self.relu)
            rep.append(SeparableConv2d(
                inplanes=inplanes,
                planes=planes,
                kernel_size=3,
                stride=1,
                dilation=dilation,
                norm_layer=norm_layer
            ))
            rep.append(norm_layer(planes))
            filters = planes

        for i in range(reps - 1):
            rep.append(self.relu)
            rep.append(SeparableConv2d(
                inplanes=filters,
                planes=filters,
                kernel_size=3,
                stride=1,
                dilation=dilation,
                norm_layer=norm_layer
            ))
            rep.append(norm_layer(filters))

        if not grow_fist:
            rep.append(self.relu)
            rep.append(SeparableConv2d(
                inplanes=inplanes,
                planes=planes,
                kernel_size=3,
                stride=1,
                dilation=dilation,
                norm_layer=norm_layer
            ))
            rep.append(norm_layer(planes))

        if stride != 1:
            rep.append(self.relu)
            rep.append(SeparableConv2d(
                inplanes=planes,
                planes=planes,
                kernel_size=3,
                stride=2,
                norm_layer=norm_layer
            ))
            rep.append(norm_layer(planes))

        if stride == 1 and is_last:
            rep.append(self.relu)
            rep.append(SeparableConv2d(
                inplanes=planes,
                planes=planes,
                kernel_size=3,
                stride=1,
                norm_layer=norm_layer
            ))
            rep.append(norm_layer(planes))

        if not start_with_relu:
            rep = rep[1:]

        self.rep = nn.Sequential(*rep)

    def forward(self, x):
        out = self.rep(x)

        if self.skip is not None:
            skip = self.skip(x)
            skip = self.skipbn(skip)
        else:
            skip = x

        out += skip
        return out


class Xception(nn.Module):
    def __init__(self, output_stride, norm_layer=None):
        """
        构建Xception网络
        @param output_stride:
        @param norm_layer:
        """
        super(Xception, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        elif norm_layer is 'gn':
            norm_layer = GroupNorm
        elif norm_layer is 'frn':
            norm_layer = FilterResponseNorm2d

        if output_stride == 16:
            entry_block3_stride = 2
            middle_block_dilation = 1
            exit_block_dilation = (1, 2)

        elif output_stride == 8:
            entry_block3_stride = 1
            middle_block_dilation = 2
            exit_block_dilation = (2, 4)
        else:
            raise NotImplementedError

        # Entry flow
        self.conv1 = nn.Conv2d(
            in_channels=3,
            out_channels=32,
            kernel_size=3,
            stride=2,
            padding=1,
            bias=False
        )
        self.bn1 = norm_layer(32)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(
            in_channels=32,
            out_channels=64,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False
        )
        self.bn2 = norm_layer(64)

        self.block1 = Block(
            inplanes=64,
            planes=128,
            reps=2,
            stride=2,
            norm_layer=norm_layer,
            start_with_relu=False,
            grow_fist=True
        )

        self.block2 = Block(
            inplanes=128,
            planes=256,
            reps=2,
            stride=2,
            norm_layer=norm_layer,
            start_with_relu=False,
            grow_fist=True
        )

        self.block3 = Block(
            inplanes=256,
            planes=728,
            reps=2,
            stride=entry_block3_stride,
            norm_layer=norm_layer,
            start_with_relu=True,
            grow_fist=True,
            is_last=True
        )

        # Middle flow
        self.block4 = Block(
            inplanes=728,
            planes=728,
            reps=3,
            stride=1,
            dilation=middle_block_dilation,
            norm_layer=norm_layer,
            start_with_relu=True,
            grow_fist=True
        )

        self.block5 = Block(
            inplanes=728,
            planes=728,
            reps=3,
            stride=1,
            dilation=middle_block_dilation,
            norm_layer=norm_layer,
            start_with_relu=True,
            grow_fist=True
        )

        self.block6 = Block(
            inplanes=728,
            planes=728,
            reps=3,
            stride=1,
            dilation=middle_block_dilation,
            norm_layer=norm_layer,
            start_with_relu=True,
            grow_fist=True
        )

        self.block7 = Block(
            inplanes=728,
            planes=728,
            reps=3,
            stride=1,
            dilation=middle_block_dilation,
            norm_layer=norm_layer,
            start_with_relu=True,
            grow_fist=True
        )

        self.block8 = Block(
            inplanes=728,
            planes=728,
            reps=3,
            stride=1,
            dilation=middle_block_dilation,
            norm_layer=norm_layer,
            start_with_relu=True,
            grow_fist=True
        )

        self.block9 = Block(
            inplanes=728,
            planes=728,
            reps=3,
            stride=1,
            dilation=middle_block_dilation,
            norm_layer=norm_layer,
            start_with_relu=True,
            grow_fist=True
        )

        self.block10 = Block(
            inplanes=728,
            planes=728,
            reps=3,
            stride=1,
            dilation=middle_block_dilation,
            norm_layer=norm_layer,
            start_with_relu=True,
            grow_fist=True
        )

        self.block11 = Block(
            inplanes=728,
            planes=728,
            reps=3,
            stride=1,
            dilation=middle_block_dilation,
            norm_layer=norm_layer,
            start_with_relu=True,
            grow_fist=True
        )

        self.block12 = Block(
            inplanes=728,
            planes=728,
            reps=3,
            stride=1,
            dilation=middle_block_dilation,
            norm_layer=norm_layer,
            start_with_relu=True,
            grow_fist=True
        )

        self.block13 = Block(
            inplanes=728,
            planes=728,
            reps=3,
            stride=1,
            dilation=middle_block_dilation,
            norm_layer=norm_layer,
            start_with_relu=True,
            grow_fist=True
        )

        self.block14 = Block(
            inplanes=728,
            planes=728,
            reps=3,
            stride=1,
            dilation=middle_block_dilation,
            norm_layer=norm_layer,
            start_with_relu=True,
            grow_fist=True
        )

        self.block15 = Block(
            inplanes=728,
            planes=728,
            reps=3,
            stride=1,
            dilation=middle_block_dilation,
            norm_layer=norm_layer,
            start_with_relu=True,
            grow_fist=True
        )

        self.block16 = Block(
            inplanes=728,
            planes=728,
            reps=3,
            stride=1,
            dilation=middle_block_dilation,
            norm_layer=norm_layer,
            start_with_relu=True,
            grow_fist=True
        )

        self.block17 = Block(
            inplanes=728,
            planes=728,
            reps=3,
            stride=1,
            dilation=middle_block_dilation,
            norm_layer=norm_layer,
            start_with_relu=True,
            grow_fist=True
        )

        self.block18 = Block(
            inplanes=728,
            planes=728,
            reps=3,
            stride=1,
            dilation=middle_block_dilation,
            norm_layer=norm_layer,
            start_with_relu=True,
            grow_fist=True
        )

        self.block19 = Block(
            inplanes=728,
            planes=728,
            reps=3,
            stride=1,
            dilation=middle_block_dilation,
            norm_layer=norm_layer,
            start_with_relu=True,
            grow_fist=True
        )

        # Exit flow
        self.block20 = Block(
            inplanes=728,
            planes=1024,
            reps=2,
            stride=1,
            dilation=exit_block_dilation[0],
            norm_layer=norm_layer,
            start_with_relu=True,
            grow_fist=False,
            is_last=True
        )

        self.conv3 = SeparableConv2d(
            inplanes=1024,
            planes=1536,
            kernel_size=3,
            stride=1,
            dilation=exit_block_dilation[1],
            norm_layer=norm_layer
        )
        self.bn3 = norm_layer(1536)

        self.conv4 = SeparableConv2d(
            inplanes=1536,
            planes=1536,
            kernel_size=3,
            stride=1,
            dilation=exit_block_dilation[1],
            norm_layer=norm_layer
        )
        self.bn4 = norm_layer(1536)

        self.conv5 = SeparableConv2d(
            inplanes=1536,
            planes=2048,
            kernel_size=3,
            stride=1,
            dilation=exit_block_dilation[1],
            norm_layer=norm_layer
        )
        self.bn5 = norm_layer(2048)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, GroupNorm, FilterResponseNorm2d)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # Entry flow
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)

        x = self.block1(x)
        x = self.relu(x)
        low_level_feat = x
        x = self.block2(x)
        x = self.block3(x)

        # Middle flow
        x = self.block4(x)
        x = self.block5(x)
        x = self.block6(x)
        x = self.block7(x)
        x = self.block8(x)
        x = self.block9(x)
        x = self.block10(x)
        x = self.block11(x)
        x = self.block12(x)
        x = self.block13(x)
        x = self.block14(x)
        x = self.block15(x)
        x = self.block16(x)
        x = self.block17(x)
        x = self.block18(x)
        x = self.block19(x)

        # Exit flow
        x = self.block20(x)
        x = self.relu(x)
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu(x)

        x = self.conv4(x)
        x = self.bn4(x)
        x = self.relu(x)

        x = self.conv5(x)
        x = self.bn5(x)
        x = self.relu(x)

        return x, low_level_feat


def xception(output_stride=16, norm_layer=None, **kwargs):
    model = Xception(output_stride=output_stride, norm_layer=norm_layer, **kwargs)
    return model

if __name__ == '__main__':
    from torchsummary import summary
    import tensorwatch as tw
    from torchviz import make_dot
    model = xception()
    x = torch.rand(1, 3, 256, 256)
    y = model(x)
    g = make_dot(y)

    g.render('xception', view=False)
    # summary(model.cuda(), (3, 256, 256))
