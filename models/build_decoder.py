#!usr/bin/python
# -*- encoding: utf-8 -*-
"""
@Time           : 2020/4/18 下午2:27
@User           : kang
@Author         : BiKang Peng
@ProjectName    : Lane_Segmentation_torch
@FileName       : build_decoder.py 
@Software       : PyCharm   
"""
from models.decoder import Decoder

def build_decoder(num_classes=8, backbone='resnet', norm_layer=None):
    model = Decoder(num_classes=num_classes, backbone=backbone, norm_layer=norm_layer)
    return model