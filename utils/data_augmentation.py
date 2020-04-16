#!usr/bin/python
# -*- encoding: utf-8 -*-
"""
@Time           : 2020/4/12 上午10:52
@User           : kang
@Author         : BiKang Peng
@ProjectName    : Lane_Segmentation_torch
@FileName       : data_augmentation.py 
@Software       : PyCharm   
"""

import torch
import numpy as np
from matplotlib import pyplot as plt
from imgaug import augmenters as iaa

sometimes = lambda aug: iaa.Sometimes(0.5, aug)  # 建立lambda表达式，


def visualize(image, mask):
    """

    @param image:
    @param mask:
    @return:
    """
    fontsize = 18
    f, ax = plt.subplots(2, 1, figsize=(8, 8))
    ax[0].imshow(image)
    ax[1].imshow(mask)
    plt.show()


class CutOut(object):
    def __init__(self, mask_size, p):
        self.mask_size = mask_size
        self.p = p

    def __call__(self, sample):
        image, mask = sample
        mask_size_half = self.mask_size // 2
        offset = 1 if self.mask_size % 2 == 0 else 0

        h, w = image.shape[:2]
        xmin_center, xmax_center = mask_size_half, w + offset - mask_size_half
        ymin_center, ymax_center = mask_size_half, h + offset - mask_size_half
        x_center = np.random.randint(xmin_center, xmax_center)
        y_center = np.random.randint(ymin_center, ymax_center)
        xmin, ymin = x_center - mask_size_half, y_center - mask_size_half
        xmax, ymax = xmin + self.mask_size, ymin + self.mask_size
        xmin, ymin, xmax, ymax = max(0, xmin), max(0, ymin), min(w, xmax), min(h, ymax)
        if np.random.uniform(0, 1) < self.p:
            image[ymin:ymax, xmin:xmax] = (0, 0, 0)
        return image, mask


class ImageAug(object):
    """
    基于像素的图像增强
    """

    def __call__(self, sample):
        image, mask = sample
        if np.random.uniform(0, 1) > 0.5:
            seq = iaa.Sequential(
                [
                    iaa.SomeOf(
                        (1, None),
                        [
                            iaa.AdditiveGaussianNoise(scale=(0, 0.2 * 255)),  # 添加高斯噪音
                            iaa.SomeOf(
                                (1, None),
                                [
                                    iaa.Sharpen(alpha=(0.1, 0.3), lightness=(0.7, 1.3)),  # 锐化
                                    iaa.Emboss(alpha=(0, 1.0), strength=(0, 2.0)),  # 浮雕效果
                                    iaa.Invert(0.05, per_channel=True),  # 5%的概率反转像素的强度，即原来的强度为v那么现在的就是255-v
                                ],
                                random_order=True  # 随机的顺序把这些操作用在图像上
                            ),
                            # 用高斯模糊，均值模糊，中值模糊中的一种增强。注意OneOf的用法
                            iaa.OneOf(
                                [
                                    iaa.GaussianBlur(sigma=(0, 1.0)),  # 高斯模糊
                                    iaa.MedianBlur(k=(3, 11)),  # 中值模糊
                                    iaa.AverageBlur(k=(2, 7)),  # 均值模糊。
                                ]
                            ),
                            iaa.OneOf(
                                [
                                    iaa.Multiply((0.5, 1.5), per_channel=0.5),  # 像素乘上0.5或者1.5之间的数字.
                                    iaa.Add((-10, 10), per_channel=0.5),  # 每个像素随机加减-10到10之间的数
                                ]
                            ),
                            sometimes(
                                iaa.Superpixels(p_replace=(0, 1.0), n_segments=(20, 200)),  # 超像素的表示
                            ),
                            sometimes(
                                # 把像素移动到周围的地方。这个方法在mnist数据集增强中有见到
                                iaa.ElasticTransformation(alpha=(0.5, 3.5), sigma=0.25)
                            ),
                            iaa.Grayscale(alpha=(0.0, 1.0)),  # 将RGB变成灰度图然后乘alpha加在原图上
                        ],
                        random_order=True  # 随机的顺序把这些操作用在图像上
                    ),
                ]
            )
            image = seq.augment_image(image)
        return image, mask


class DeformAug(object):
    """
    基于形态的数据增强
    """

    def __call__(self, sample):
        image, mask = sample
        if np.random.uniform(0, 1) > 0.5:
            seq = iaa.Sequential(
                [
                    iaa.CropAndPad(percent=(-0.05, 0.1))
                ]
            )
            # 固定变换
            seg_to = seq.to_deterministic()
            image = seg_to.augment_image(image)
            mask = seg_to.augment_image(mask)
        return image, mask


if __name__ == '__main__':
    from tqdm import tqdm
    import torch
    import numpy as np
    import matplotlib.pyplot as plt
    from torchvision import transforms
    from torch.utils.data import DataLoader
    from utils.image_precess import LaneDataset, ToTensor

    kwargs = {'num_workers': 4, 'pin_memory': True} if torch.cuda.is_available() else {}
    train_dataset = LaneDataset(
        csv_file='/home/kang/CV-Project/Lane_Segmentation_torch/data_list/train.csv',
        transform=transforms.Compose(
            [
                ImageAug(),
                DeformAug(),
                CutOut(32, 0.5),
                ToTensor()
            ]
        )
    )
    training_data_batch = DataLoader(train_dataset, batch_size=2, shuffle=True, drop_last=True, **kwargs)
    dataprocess = tqdm(training_data_batch)
    for batch_item in dataprocess:
        image, mask = batch_item['image'], batch_item['mask']
        if torch.cuda.is_available():
            image, mask = image.cuda(), mask.cuda()
