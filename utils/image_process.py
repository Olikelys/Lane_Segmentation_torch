#!usr/bin/python
# -*- encoding: utf-8 -*-
"""
@Time           : 2020/4/15 下午4:41
@User           : kang
@Author         : BiKang Peng
@ProjectName    : Lane_Segmentation_torch
@FileName       : image_process.py
@Software       : PyCharm   
"""

import os
import cv2
import random
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from utils.process_labels import encode_labels, decode_labels, \
    decode_color_labels


def crop_resize_data(image, label=None, image_size=(1024, 384), offset=690):
    """
    裁剪图像以丢弃无用的部分
    h, w, c = image.shape
    cv2.resize(image,(w,h))
    @param image: 图像
    @param label: 标签
    @param image_size: 图像大小
    @param offset: 裁剪值
    @return:
    """
    roi_image = image[offset:, :]
    if label is not None:
        roi_label = label[offset:, :]
        # INTER_LINEAR 双线性插值
        train_image = cv2.resize(roi_image, image_size, interpolation=cv2.INTER_LINEAR)
        # INTER_NEAREST 最近领插值
        train_label = cv2.resize(roi_label, image_size, interpolation=cv2.INTER_NEAREST)
        return train_image, train_label
    else:
        train_image = cv2.resize(roi_image, image_size, interpolation=cv2.INTER_LINEAR)
        return train_image


class LaneDataset(Dataset):
    def __init__(self, csv_file, transform=None):
        """
        定义车道线数据集类
        @param csv_file:
        @param transform:
        """
        super(LaneDataset, self).__init__()
        self.data = pd.read_csv(csv_file, header=None, names=['image', 'label'])
        self.images = self.data['image'].values[1:]
        self.labels = self.data['label'].values[1:]
        self.transform = transform

    def __len__(self):
        return self.labels.shape[0]

    def __getitem__(self, idx):
        ori_image = cv2.imread(self.images[idx])
        ori_mask = cv2.imread(self.labels[idx], cv2.IMREAD_GRAYSCALE)
        train_img, train_mask = crop_resize_data(ori_image, ori_mask)
        train_mask = encode_labels(train_mask)
        sample = [train_img.copy(), train_mask.copy()]
        if self.transform:
            sample = self.transform(sample)
        return sample

class ToTensor(object):
    def __call__(self, sample):
        image, mask = sample
        image = np.transpose(image, (2, 0, 1))
        image = image.astype(np.float32)
        mask = mask.astype(np.long)
        return {
            'image': torch.from_numpy(image.copy()),
            'mask': torch.from_numpy(mask.copy())
        }

def expend_resize_data(prediction=None, submission_size=(3384, 1710), offset=690):
    pred_mask = decode_labels(prediction)
    expend_mask = cv2.resize(pred_mask, (submission_size[0], submission_size[1]-offset), interpolation=cv2.INTER_NEAREST)
    submission_mask = np.zeros((submission_size[1], submission_size[0]), dtype='uint8')
    submission_mask[offset:, :] = expend_mask
    return submission_mask

def expend_resize_color_data(prediction=None, submission_size=(3384, 1710), offset=690):
    color_pred_mask  = decode_color_labels(prediction)
    color_pred_mask = np.transpose(color_pred_mask, (1, 2, 0))
    color_expend_mask = cv2.resize(color_pred_mask, (submission_size[0], submission_size[1]-offset),
                                   interpolation=cv2.INTER_NEAREST)
    color_submission_mask = np.zeros((submission_size[1], submission_size[0], 3), dtype='uint8')
    color_submission_mask[offset:, :, :] = color_expend_mask
    return color_submission_mask