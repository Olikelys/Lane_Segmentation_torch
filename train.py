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
import os
from torchvision import transforms
from sklearn.model_selection import ShuffleSplit
from torch.utils.data import DataLoader
from models.deeplab_v3p import DeepLabV3p
from utils.loss import FocalLoss, FocalTversky_loss
from utils.metric import Evaluator
from utils.image_process import LaneDataset, ToTensor
from utils.data_augmentation import ImageAug, DeformAug, CutOut
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
    train_dataset = LaneDataset(
        csv_file=cfg.TRAIN_CSV_FILE,
        transform=transforms.Compose(
            [
                ImageAug(),
                DeformAug(),
                CutOut(64, 0.5),
                ToTensor()
            ]
        )
    )
    train_dataloader = DataLoader(train_dataset, batch_size=cfg.BATCHES, shuffle=cfg.TRAIN_SHUFFLE,
                                  num_workers=cfg.DATA_WORKERS, drop_last=True)
    val_dataset = LaneDataset(
        csv_file=cfg.VAL_CSV_FILE,
        transform=transforms.Compose(
            [
                ToTensor()
            ]
        )
    )
    val_dataloader = DataLoader(val_dataset, batch_size=cfg.BATCHES, shuffle=cfg.VAL_TEST_SHUFFLE,
                                num_workers=cfg.DATA_WORKERS)

    # 模型构建
    model = DeepLabV3p()
    model = model.to(device)

    # 损失函数和优化器
    if cfg.LOSS == 'ce':
        criterion = nn.CrossEntropyLoss().to(device)
    elif cfg.LOSS == 'focal':
        criterion = FocalLoss().to(device)
    elif cfg.LOSS == 'focalTversky':
        criterion = FocalTversky_loss().to(device)

    optimizer = opt.Adam(model.parameters(), lr=cfg.TRAIN_LR)

    for epo in range(epoch):
        # 训练部分
        train_loss = 0
        model.train()
        for index, batch_item in enumerate(train_dataloader):
            image, mask = batch_item['image'].to(device), batch_item['mask'].to(device)
            optimizer.zero_grad()
            output = model(image)
            loss = criterion(output, mask)
            loss.backward()
            # 取出loss数值
            iter_loss = loss.item()
            train_loss += loss
            optimizer.step()

            if np.mod(index, 8) == 0:
                line = 'epoch {}, {}/{}, train loss is {}'.format(epo, index, len(train_dataloader), iter_loss)
                print(line)
                with open(os.path.join(cfg.LOG_DIR, 'log.txt'), 'a') as f:
                    f.write(line)
                    f.write('\r\n')

        #验证部分
        val_loss = 0
        model.eval()
        with torch.no_grad():
            for index, batch_item in enumerate(val_dataloader):
                image, mask = batch_item['image'].to(device), batch_item['mask'].to(device)

                optimizer.zero_grad()
                output = model(image)
                loss = criterion(output, mask)
                iter_loss = loss.item()
                val_loss += iter_loss

                # 记录相关指标
                pred = output.cpu().numpy()
                mask = mask.cpu().numpy()
                pred = np.argmax(pred, axis=1)
                evaluator.add_batch(mask, pred)

        line_epoch = 'epoch train loss = %.3f, epoch val loss = %.3f' % (train_loss/len(train_dataloader),
                                                                         val_loss/len(val_dataloader))
        print(line_epoch)
        with open(os.path.join(cfg.LOG_DIR, 'log.txt'), 'a') as f:
            f.write(line)
            f.write('\r\n')

        ACC = evaluator.Pixel_Accuracy()
        mIoU = evaluator.Mean_Intersection_over_Union()

        # tensorboard记录
        writer.add_scalar('train_loss', train_loss/len(train_dataloader), epo)
        writer.add_scalar('val_loss', val_loss/len(val_dataloader), epo)
        writer.add_scalar('Acc', ACC, epo)
        writer.add_scalar('mIoU', mIoU, epo)

        # 每次验证，根据新得出的mIoU指标来保存模型
        new_pred = mIoU
        if new_pred > best_pred:
            best_pred = new_pred
            save_path = os.path.join(cfg.MODEL_SAVE_DIR, '{}_{}_{}_{}_{}.pth'.format(cfg.BACKBONE, cfg.LAYERS,
                                                                                        cfg.NORM_LAYER, cfg.LOSS,
                                                                                        epo))

            torch.save(model.state_dict(), save_path)

if __name__ == '__main__':
    train(epoch=300)


