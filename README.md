# **Lane_Segmentation_torch**
**无人车车道线检测项目，基于Pytorch 1.0 以上版本开发**

------

[TOC]

## 项目架构

### Data(数据处理)

#### **<u>[utils文件夹] [utils](utils) ()</u>**

| 文件名               | 内容                                                         |
| -------------------- | ------------------------------------------------------------ |
| make_list.py         | 生成图像数据与标签数据相对应的CSV文件                        |
| process_labels.py    | 对标签进行处理                                               |
| image_process.py     | 对图像进行处理                                               |
| data_augmentation.py | 数据增强的实现                                               |
| generate_dataset.py  | 生成训练、验证及测试数据                                     |
| metric.py            | 评估函数，包含global_accuracy, class_accuracies, precision_score, recall_score, f1_score, miou |

### Model(模型构建)

#### **<u>[bn文件夹] [bn](bn) ()</u>**

| 文件名 | 内容                    |
| ------ | ----------------------- |
| frn.py | FilterResponseNorm 实现 |
| gn.py  | GroupNorm 实现          |

#### **<u>[loss文件夹] [loss](loss) ()</u>**

| 文件名               | 内容                   |
| -------------------- | ---------------------- |
| Focal_loss.py        | Focal_loss 实现        |
| FocalTversky_loss.py | FocalTversky_loss 实现 |
| DC_and_CE_loss.py    | DC_and_CE_loss 实现    |

#### **<u>[models文件夹] [models](models) ()</u>**

| 文件名       | 内容                                                         |
| ------------ | ------------------------------------------------------------ |
| resnet.py    | [resnet](https://arxiv.org/abs/1512.03385) 实现              |
| resgroup.py  | resgroup 实现                                                |
| iresnet.py   | [iresnet](https://arxiv.org/abs/2004.04989) 实现（最新的resnet进阶版） |
| iresgroup.py | iresgroup 实现                                               |
| xception.py  | [xception](https://arxiv.org/abs/1610.02357) 实现            |
| assp.py      | deeplab-v3+ 的ASPP层实现                                     |
| decoder.py   | deeplab-v3+ 解码层实现                                       |
| deeplab-v3p  | [deeplab-v3+](https://arxiv.org/abs/1802.02611) 实现         |

### Training(模型训练)

### Inference(模型评估)

### Deployment(模型部署)

### 



