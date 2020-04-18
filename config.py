#!usr/bin/python
# -*- encoding: utf-8 -*-
"""
@Time           : 2020/4/18 下午2:55
@User           : kang
@Author         : BiKang Peng
@ProjectName    : Lane_Segmentation_torch
@FileName       : config.py 
@Software       : PyCharm   
"""

import os


class Configuration(object):
    """
    定义项目所需超参数
    """

    def __init__(self):
        """"""
        '''
        定义构建网络参数
        '''
        # 所使用的骨干网络：resnet, resgroup, iresnet, iresgroup, xception
        self.BACKBONE = 'iresnet'
        # 所使用的骨干网路的层数：50, 101, 152, 200, 302, 404, 1001
        self.LAYERS = 50
        # 下采样倍数
        self.OUTPUT_STRIDE = 16
        # 分类数
        self.NUM_CLASSES = 8
        # 选择BN
        self.NORM_LAYER = None
        # 重置bn为0
        self.FREEZE_BN = False

        '''
        定义训练所需参数
        '''
        # 根目录
        self.ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname("__file__")))
        # 训练集路径
        self.TRAIN_CSV_FILE = os.path.join(self.ROOT_DIR, 'data_list', 'train.csv')
        # 验证集路径
        self.VAL_CSV_FILE = os.path.join(self.ROOT_DIR, 'data_list', 'val.csv')
        # 测试集路径
        self.TEST_CSV_FILE = os.path.join(self.ROOT_DIR, 'data_list', 'test.csv')
        # 定义当前是训练还是验证或测试状态
        self.TRAIN = 'train'
        self.VAL = 'val'
        self.TEST = 'test'
        # 采用哪几种图像增强算法
        self.AUG = 'all'
        # 每次输入的图片数
        self.TRAIN_BATCHES = 2
        # 是否打乱
        self.TRAIN_SHUFFLE = True
        # 使用进程（并行处理）
        self.DATA_WORKERS = 4
        # 保存日志文件路径
        self.LOG_DIR = os.path.join(self.ROOT_DIR, 'logs', self.BACKBONE + str(self.LAYERS))
        # 使用GPU数量
        self.TRAIN_GPUS = 1
        # 与训练权重路径
        self.TRAIN_CKPT = ''
        # 训练学习率
        self.TRAIN_LR = 0.007
        # 带动量的SGD的动量值
        self.TRAIN_MOMENTUM = 0.9
        # 训练论次
        self.TRAIN_MINEPOCH = 0
        self.TRAIN_EPOCHS = 100
        # 权重保存路径
        self.MODEL_SAVE_DIR = os.path.join(self.ROOT_DIR, 'weights', self.BACKBONE + str(self.LAYERS))
        # 学习率衰减系数
        self.TRAIN_POWER = 0.9
        # 检查
        self.__check()

    def __check(self):
        if not torch.cuda.is_available():
            raise ValueError('config.py: cuda is not available')
        if self.TRAIN_GPUS == 0:
            raise ValueError('config.py: the number of GPU is 0')
        if self.TRAIN_GPUS != torch.cuda.device_count():
            raise ValueError('config.py: GPU number is not matched')
        if not os.path.isdir(self.LOG_DIR):
            os.makedirs(self.LOG_DIR)
        if not os.path.isdir(self.MODEL_SAVE_DIR):
            os.makedirs(self.MODEL_SAVE_DIR)


cfg = Configuration()
