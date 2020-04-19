#!usr/bin/python
# -*- encoding: utf-8 -*-
"""
@Time           : 2020/4/18 上午9:55
@User           : kang
@Author         : BiKang Peng
@ProjectName    : Lane_Segmentation_torch
@FileName       : build_backbone.py 
@Software       : PyCharm   
"""

import models.resnet as resnet
import models.iresnet as iresnet
import models.resgroup as resgroup
import models.iresgroup as iresgroup
import models.xception as xception


def build_backbone(backbone='resnet-50', layers=50, output_stride=16, norm_layer=None):
    # if norm_layer is None:
    #     norm_layer = nn.BatchNorm2d
    # elif norm_layer is 'gn':
    #     norm_layer = GroupNorm
    # elif norm_layer is 'frn':
    #     norm_layer = FilterResponseNorm2d
    if backbone is 'resnet':
        if layers == 50:
            model = resnet.resnet50(norm_layer=norm_layer)
            return model
        elif layers == 101:
            model = resnet.resnet101(norm_layer=norm_layer)
            return model
        elif layers == 152:
            model = resnet.resnet152(norm_layer=norm_layer)
            return model
        elif layers == 200:
            model = resnet.resnet200(norm_layer=norm_layer)
            return model

    elif backbone is 'resgroup':
        if layers == 50:
            model = resgroup.resgroup50(norm_layer=norm_layer)
            return model
        elif layers == 101:
            model = resgroup.resgroup101(norm_layer=norm_layer)
            return model
        elif layers == 152:
            model = resgroup.resgroup152(norm_layer=norm_layer)
            return model

    elif backbone is 'iresnet':
        if layers == 50:
            model = iresnet.iresnet50(norm_layer=norm_layer)
            return model
        elif layers == 101:
            model = iresnet.iresnet101(norm_layer=norm_layer)
            return model
        elif layers == 152:
            model = iresnet.iresnet152(norm_layer=norm_layer)
            return model
        elif layers == 200:
            model = iresnet.iresnet200(norm_layer=norm_layer)
            return model
        elif layers == 302:
            model = iresnet.iresnet302(norm_layer=norm_layer)
            return model
        elif layers == 404:
            model = iresnet.iresnet404(norm_layer=norm_layer)
            return model
        elif layers == 1001:
            model = iresnet.iresnet1001(norm_layer=norm_layer)
            return model

    elif backbone is 'iresgroup-50':
        if layers == 50:
            model = iresgroup.iresgroup50(norm_layer=norm_layer)
            return model
        elif layers == 101:
            model = iresgroup.iresgroup101(norm_layer=norm_layer)
            return model
        elif layers == 152:
            model = iresgroup.iresgroup152(norm_layer=norm_layer)
            return model

    elif backbone is 'xception':
        model = xception.xception(output_stride=output_stride, norm_layer=norm_layer)
        return model
