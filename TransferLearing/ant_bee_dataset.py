"""アリとハチの画像を分類するためのデータセット
"""
import os
import sys

# os.sepはプラットフォーム固有の区切り文字(Windows: `\`, Unix: `/`)
module_parent_dir = os.sep.join([os.path.dirname(__file__), '..'])
# print("module_parent_dir", module_parent_dir)
sys.path.append(module_parent_dir)

from log_conf import logging
log = logging.getLogger(__name__)
# log.setLevel(logging.WARN)
# log.setLevel(logging.INFO)
log.setLevel(logging.DEBUG)

import argparse
import datetime
import json
import glob
import random
import os.path as osp
from tqdm import tqdm


from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision import models, transforms
import torch.optim as optim
import torch.utils.data as data
#torch.backends.cudnn.deterministic = True
#torch.backends.cudnn.benchmark = False

from type_hint import *
from .preprocess import ImageTransform


# アリとハチの画像へのファイルパスのリストを作成する
def make_datapath_list(phase : str = 'train'):

    rootpath = f"{os.path.dirname(__file__)}/data/hymenoptera_data/"
    target_path = osp.join(rootpath + phase + '/**/*.jpg')
    # print(target_path)

    # ファイルパスを取得
    path_list = []
    for path in glob.glob(target_path):
        path_list.append(path)

    # for i, path in enumerate(path_list):
    #     print(str(i) + ": " +  path)

    return path_list


# アリとハチの画像データセット
class HymenopteraDataset(data.Dataset):

    def __init__(self, 
                 file_list : List[str], 
                 transform : Optional[ImageTransform] = None, 
                 phase : str = 'train',
                 ):
        self.file_list = file_list
        self.transform = transform
        self.phase = phase
    
    # 必須 1)
    def __len__(self):
        return len(self.file_list)

    # 必須 2)
    def __getitem__(self, index):

        img_path = self.file_list[index]
        # print('img_path:', img_path)
        img = Image.open(img_path)

        img_transformed = self.transform(img, self.phase) # torch.Size([3, 224, 224])

        if self.phase == 'train':
            label = img_path[30:34]
        elif self.phase == 'val':
            label = img_path[28:32]

        label = img_path.split(os.sep)[-2] # ants / bees

        label_t = torch.zeros((2,), dtype=torch.float32)
        if label == 'ants':
            label_t[0] = 1
        elif label == 'bees':
            label_t[1] = 1
        else:
            raise ValueError(f'{label} is invalid.')

        return img_transformed, label_t
    

# データセット作成
def make_hymenoptera_dataset() -> Tuple[HymenopteraDataset, HymenopteraDataset]:

    # 訓練
    train_list = make_datapath_list(phase='train')

    # 検証
    val_list = make_datapath_list(phase='val')

    # データセット
    resize = 224
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)
    transform = ImageTransform(resize, mean, std)
    train_dataset = HymenopteraDataset(file_list=train_list, transform=transform, phase='train')
    val_dataset = HymenopteraDataset(file_list=val_list, transform=transform, phase='val')
    
    return train_dataset, val_dataset



class HymenopteraDatasetApp:
    def __init__(self, sys_argv : Optional[Any] = None):
        if sys_argv is None:
            sys_argv = sys.argv[1:]

        parser : Any = argparse.ArgumentParser()

        # 必要であれば, ここにアプリ引数を登録

        self.cli_args = parser.parse_args(sys_argv)
        self.time_str = datetime.datetime.now().strftime('%Y-%m-%d_%H.%M.%S')

    def main(self):
        """アプリケーション
        """
        log.info("Starting {}, {}".format(type(self).__name__, self.cli_args))

        # 訓練データのパス一覧
        trn_file_list = make_datapath_list()
        print('Hymenoptera file list(train):\n', trn_file_list)

        trn_ds, val_ds = make_hymenoptera_dataset()

        print("Hymenoptera training dataset:\n", trn_ds)
        print("Hymenoptera validation dataset:\n", val_ds)




