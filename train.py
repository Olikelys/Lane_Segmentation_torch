#!usr/bin/python
# -*- encoding: utf-8 -*-
"""
@Time           : 2020/4/18 下午2:52
@User           : kang
@Author         : BiKang Peng
@ProjectName    : Lane_Segmentation_torch
@FileName       : train.py 
@Software       : PyCharm   
"""
import torch
import torch.nn as nn
import torch.optim as opt
import numpy as np
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
from models.deeplab_v3p import DeepLabV3p
from utils.loss import FocalLoss, FocalTversky_loss
from utils.generate_dataset import generate_dataset
from utils.metric import Evaluator
from config import cfg

def train(epoch=400):
    # 创建指标计算对象
    evaluator = Evaluator(8)

    # 定义好最好的指标miou数值， 初始化为0
    best_pred = 0.0

    # 写入日志
    writer = SummaryWriter(cfg.LOG_DIR)

    # 指定GPU
    device = torch.device(0)

    # 创建数据
    train_dataset = generate_dataset(cfg.TRAIN_CSV_FILE, cfg.TRAIN, cfg.AUG)
    train_dataloader = DataLoader(train_dataset, batch_size=cfg.BATCHES, shuffle=cfg.TRAIN_SHUFFLE,
                                  num_workers=cfg.DATA_WORKERS, drop_last=True)
    val_dataset = generate_dataset(cfg.VAL_CSV_FILE, cfg.VAL)
    val_dataloader = DataLoader(val_dataset, batch_size=cfg.BATCHES, shuffle=cfg.VAL_TEST_SHUFFLE,
                                num_workers=cfg.DATA_WORKERS)

    # 模型构建
    model = DeepLabV3p()
    model = model.to(device)

    # 损失函数和优化器
    if cfg.LOSS == 'ce':
        loss = nn.CrossEntropyLoss().to(device)
    elif cfg.LOSS == 'focal':
        loss = FocalLoss().to(device)
    elif cfg.LOSS == 'focalTversky':
        loss = FocalTversky_loss().to(device)

    optimizer = opt.Adam(model.parameters(), lr=cfg.TRAIN_LR)








