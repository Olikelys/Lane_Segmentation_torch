#!usr/bin/python
# -*- encoding: utf-8 -*-
"""
@Time           : 2020/4/17 上午10:05
@User           : kang
@Author         : BiKang Peng
@ProjectName    : Lane_Segmentation_torch
@FileName       : nd_softmax.py 
@Software       : PyCharm   
"""

import torch
def softmax_helper(x):
    rpt = [1 for _ in range(len(x.size()))]
    rpt[1] = x.size(1)
    x_max = x.max(1, keepdim=True)[0].repeat(*rpt)
    e_x = torch.exp(x - x_max)
    return e_x / e_x.sum(1, keepdim=True).repeat(*rpt)