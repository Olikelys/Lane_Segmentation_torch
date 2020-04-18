#!usr/bin/python
# -*- encoding: utf-8 -*-
"""
@Time           : 2020/4/18 上午10:32
@User           : kang
@Author         : BiKang Peng
@ProjectName    : Lane_Segmentation_torch
@FileName       : build_aspp.py 
@Software       : PyCharm   
"""
from models.aspp import aspp
def build_aspp(backbone='resnet', output_stride=16, norm_layer=None):
    model = aspp(backbone=backbone, output_stride=output_stride, norm_layer=norm_layer)
    return model
