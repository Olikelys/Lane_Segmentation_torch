#!usr/bin/python
# -*- encoding: utf-8 -*-
"""
@Time           : 2020/4/17 上午10:07
@User           : kang
@Author         : BiKang Peng
@ProjectName    : Lane_Segmentation_torch
@FileName       : tensor_utilities.py 
@Software       : PyCharm   
"""
import numpy as np
import torch
from torch import nn

def sum_tensor(inp, axes, keepdim=False):
    if keepdim:
        for ax in axes:
            inp = inp.sum(int(ax), keepdim=True)
    else:
        for ax in sorted(axes, reverse=True):
            inp = inp.sum(int(ax))
    return inp