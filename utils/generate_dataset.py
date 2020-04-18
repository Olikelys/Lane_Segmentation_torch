#!usr/bin/python
# -*- encoding: utf-8 -*-
"""
@Time           : 2020/4/18 下午6:08
@User           : kang
@Author         : BiKang Peng
@ProjectName    : Lane_Segmentation_torch
@FileName       : generate_dataset.py 
@Software       : PyCharm   
"""
import torch
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms
from torch.utils.data import DataLoader
from utils.image_process import LaneDataset, ToTensor
from utils.data_augmentation import ImageAug, DeformAug, CutOut
import sys
sys.path.append('../')
from config import cfg



def generate_dataset(csv_file, types, aug):
    if types is 'train':
        if aug is None:
            dataset = LaneDataset(
                csv_file=csv_file,
                transform=transforms.Compose(
                    [
                        ToTensor()
                    ]
                )
            )
            return dataset
        elif aug is 'ImageAug':
            dataset = LaneDataset(
                csv_file=csv_file,
                transform=transforms.Compose(
                    [
                        ImageAug(),
                        CutOut(64, 0.5),
                        ToTensor()
                    ]
                )
            )
            return dataset
        elif aug is 'DeformAug':
            dataset = LaneDataset(
                csv_file=csv_file,
                transform=transforms.Compose(
                    [
                        DeformAug(),
                        CutOut(64, 0.5),
                        ToTensor()
                    ]
                )
            )
            return dataset
        elif aug is 'All':
            dataset = LaneDataset(
                csv_file=csv_file,
                transform=transforms.Compose(
                    [
                        ImageAug(),
                        DeformAug(),
                        CutOut(64, 0.5),
                        ToTensor()
                    ]
                )
            )
            return dataset
    elif types is 'val':
        dataset = LaneDataset(
            csv_file=csv_file,
            transform=transforms.Compose(
                [
                    ToTensor()
                ]
            )
        )
        return dataset
    elif types is 'test':
        dataset = LaneDataset(
            csv_file=csv_file,
            transform=transforms.Compose(
                [
                    ToTensor()
                ]
            )
        )
        return dataset
    else:
        raise NotImplementedError
