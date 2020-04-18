#!usr/bin/python
# -*- encoding: utf-8 -*-
"""
@Time           : 2020/4/17 下午12:06
@User           : kang
@Author         : BiKang Peng
@ProjectName    : Lane_Segmentation_torch
@FileName       : frn.py 
@Software       : PyCharm   
"""

import torch
import torch.nn as nn

__all__ = ['FilterResponseNorm1d', 'FilterResponseNorm2d',
           'FilterResponseNorm3d']


class FilterResponseNormNd(nn.Module):

    def __init__(self, ndim, num_features, eps=1e-6, learnable_eps=False):
        assert ndim in [3, 4, 5], 'FilterResponseNorm only supports 3d, 4d or 5d inputs.'
        super(FilterResponseNormNd, self).__init__()
        shape = (1, num_features) + (1,) * (ndim - 2)
        self.eps = nn.Parameter(torch.ones(*shape) * eps)
        if not learnable_eps:
            self.eps.requires_grad_(False)
        self.weight = nn.Parameter(torch.Tensor(*shape))
        self.bias = nn.Parameter(torch.Tensor(*shape))
        self.tau = nn.Parameter(torch.Tensor(*shape))
        self.reset_parameters()

    def forward(self, x):
        avg_dims = tuple(range(2, x.dim()))
        nu2 = torch.pow(x, 2).mean(dim=avg_dims, keepdim=True)
        x = x * torch.rsqrt(nu2 + torch.abs(self.eps))
        return torch.max(self.weight * x + self.bias, self.tau)

    def reset_parameters(self):
        nn.init.ones_(self.weight)
        nn.init.zeros_(self.bias)
        nn.init.zeros_(self.tau)


class FilterResponseNorm1d(FilterResponseNormNd):

    def __init__(self, num_features, eps=1e-6, learnable_eps=False):
        super(FilterResponseNorm1d, self).__init__(
            3, num_features, eps=eps, learnable_eps=learnable_eps)


class FilterResponseNorm2d(FilterResponseNormNd):

    def __init__(self, num_features, eps=1e-6, learnable_eps=False):
        super(FilterResponseNorm2d, self).__init__(
            4, num_features, eps=eps, learnable_eps=learnable_eps)


class FilterResponseNorm3d(FilterResponseNormNd):

    def __init__(self, num_features, eps=1e-6, learnable_eps=False):
        super(FilterResponseNorm3d, self).__init__(
            5, num_features, eps=eps, learnable_eps=learnable_eps)
