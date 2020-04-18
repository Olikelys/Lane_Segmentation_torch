#!usr/bin/python
# -*- encoding: utf-8 -*-
"""
@Time           : 2020/4/18 下午2:30
@User           : kang
@Author         : BiKang Peng
@ProjectName    : Lane_Segmentation_torch
@FileName       : deeplab-v3p.py 
@Software       : PyCharm   
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.build_backbone import build_backbone
from models.build_aspp import build_aspp
from models.build_decoder import build_decoder
import sys

sys.path.append('../')
from bn.gn import GroupNorm
from bn.frn import FilterResponseNorm2d
from config import cfg


class DeepLabV3p(nn.Module):
    def __init__(self, backbone=cfg.BACKBONE, layers=cfg.LAYERS, output_stride=cfg.OUTPUT_STRIDE,
                 num_classes=cfg.NUM_CLASSES, norm_layer=cfg.NORM_LAYER, freeze_bn=cfg.FREEZE_BN):
        """
        构建DeepLabV3Plus
        @param backbone:
        @param layers:
        @param output_stride:
        @param num_classes:
        @param norm_layer:
        @param freeze_bn:
        """
        super(DeepLabV3p, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        elif norm_layer is 'gn':
            norm_layer = GroupNorm
        elif norm_layer is 'frn':
            norm_layer = FilterResponseNorm2d

        self.backbone = build_backbone(backbone=backbone, layers=layers, output_stride=output_stride,
                                       norm_layer=norm_layer)
        self.aspp = build_aspp(backbone=backbone, output_stride=output_stride, norm_layer=norm_layer)
        self.decoder = build_decoder(num_classes=num_classes, backbone=backbone, norm_layer=norm_layer)

        if freeze_bn:
            self.freeze_bn()

    def forward(self, inputs):
        x, low_level_feat = self.backbone(inputs)
        x = self.aspp(x)
        x = self.decoder(x, low_level_feat)
        x = F.interpolate(x, size=inputs.size()[2:], mode='bilinear', align_corners=True)

        return x

    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()
            elif isinstance(m, GroupNorm):
                m.eval()
            elif isinstance(m, FilterResponseNorm2d):
                m.eval()

    def get_1x_lr_params(self):
        modules = [self.backbone]
        for i in range(len(modules)):
            for m in modules[i].named_modules():
                if isinstance(m[1], nn.Conv2d) or isinstance(m[1], nn.BatchNorm2d) or isinstance(m[1], GroupNorm) or \
                        isinstance(m[1], FilterResponseNorm2d):
                    for p in m[1].parameters():
                        if p.requires_grad:
                            yield p

    def get_10x_lr_params(self):
        modules = [self.aspp, self.decoder]
        for i in range(len(modules)):
            for m in modules[i].named_modules():
                if isinstance(m[1], nn.Conv2d) or isinstance(m[1], nn.BatchNorm2d) or isinstance(m[1], GroupNorm) or \
                        isinstance(m[1], FilterResponseNorm2d):
                    for p in m[1].parameters():
                        if p.requires_grad:
                            yield p


if __name__ == '__main__':
    from torchsummary import summary

    model = DeepLabV3p()
    # model.eval()
    # inputs = torch.rand(1, 3, 513, 513)
    # outputs = model(inputs)
    # print(outputs.size())
    summary(model.cuda(), (3, 224, 224))
