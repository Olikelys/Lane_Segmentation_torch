'''
@Descripttion: 把图片映射成CSV文件
@version: 
@Author: Bikang Peng
@Date: 2020-04-11 21:39:15
@LastEditors: Bikang Peng
@LastEditTime: 2020-04-11 22:50:00
'''

import os
import pandas as pd
from sklearn.utils import shuffle

# 定义image_list, lable_list两个列表
image_list = []
label_list = []

# 定义image 和 lable 两个文件夹路径
image_dir = '/home/kang/DATASETS/Self-driving_lane_detection/ColorImage/'
label_dir = '/home/kang/DATASETS/Self-driving_lane_detection/Gray_Label/'

# ColorImage
for s1 in os.listdir(image_dir):
    # ColorImage/Road02
    image_sub_dir1 = os.path.join(image_dir, s1)
    # Gray_Label/Lable_road02/Lable
    label_sub_dir1 = os.path.join(label_dir, 'Lable_' + str.lower(s1), 'Lable')

    # Road02
    for s2 in os.listdir(image_sub_dir1):
        # ColorImage/Road02/Record001
        image_sub_dir2 = os.path.join(image_sub_dir1, s2)
        # Gray_Label/Lable_road02/Lable/Record001
        label_sub_dir2 = os.path.join(label_sub_dir1, s2)

        # Record001
        for s3 in os.listdir(image_sub_dir2):
            # ColorImage/Road02/Record001/Camera5
            image_sub_dir3 = os.path.join(image_sub_dir2, s3)
            # Gray_Label/Lable_road02/Lable/Record001/Camera5
            label_sub_dir3 = os.path.join(label_sub_dir2, s3)

            # 图片 image为jpg，label为png
            for s4 in os.listdir(image_sub_dir3):
                s44 = s4.replace('.jpg', '_bin.png')
                image_sub_dir4 = os.path.join(image_sub_dir3, s4)
                label_sub_dir4 = os.path.join(label_sub_dir3, s44)

                if not os.path.exists(image_sub_dir4):
                    print(image_sub_dir4)
                    continue
                if not os.path.exists(label_sub_dir4):
                    print(label_sub_dir4)
                    continue
                image_list.append(image_sub_dir4)
                label_list.append(label_sub_dir4)

assert len(image_list) == len(label_list)
print('The length of image dataset is {}, and label is {}'.format(len(image_list), len(label_list)))
total_length = len(image_list)
sixth_part = int(total_length * 0.6)
eighth_part = int(total_length * 0.8)

all = pd.DataFrame({'image': image_list, 'label':label_list})
all_shuffle = shuffle(all)

train_dataset = all_shuffle[:sixth_part]
val_dataset = all_shuffle[sixth_part:eighth_part]
test_dataset = all_shuffle[eighth_part:]

train_dataset.to_csv('/home/kang/CV-Project/Lane_Segmentation_torch/data_list/train.csv')
val_dataset.to_csv('/home/kang/CV-Project/Lane_Segmentation_torch/data_list/val.csv')
test_dataset.to_csv('/home/kang/CV-Project/Lane_Segmentation_torch/data_list/test.csv')
