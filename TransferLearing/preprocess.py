"""画像の前処理
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


# 前処理クラス
class ImageTransform():

    def __init__(self, 
                 resize : int, 
                 mean : Tuple[float, float, float], 
                 std : Tuple[float, float, float],
                 ):
        
        self.data_transform = {
            'train': transforms.Compose([
                transforms.RandomResizedCrop(resize, scale=(0.5, 1.0)),  # ランダムクロップ
                transforms.RandomHorizontalFlip(),                       # ランダムフリップ(水平)
                transforms.ToTensor(),                                   # Tensor変換
                transforms.Normalize(mean, std)                          # 標準化
            ]),
            'val': transforms.Compose([
                transforms.Resize(resize),                               # リサイズ
                transforms.CenterCrop(resize),                           # クロップ
                transforms.ToTensor(),                                   # Tensor変換
                transforms.Normalize(mean, std)                          # 標準化
            ])
        }

    def __call__(self, img, phase='train'):
        return self.data_transform[phase](img)


class PreprocessImageApp:
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

        # 画像読み込み
        image_file_path = f"{os.path.dirname(__file__)}/data/goldenretriever-3724972_640.jpg"
        img = Image.open(image_file_path)

        # 原画像を表示
        plt.imshow(img)
        plt.show()

        # 前処理と処理後の画像を表示
        resize = 224
        mean = (0.485, 0.456, 0.406)
        std = (0.229, 0.224, 0.225)
        transform = ImageTransform(resize, mean, std)
        img_transformed = transform(img, 'train') # torch.Size([3, 224, 224])
        img_transformed_transposed = img_transformed.numpy().transpose((1,2,0))

        print("img_transformed_transposed :",  img_transformed_transposed)
        img_transformed_transposed_cliped = np.clip(img_transformed_transposed, 0, 1) # 0-1にクリップ
        print("img_transformed_transposed_cliped :",  img_transformed_transposed_cliped)
        plt.imshow(img_transformed_transposed_cliped)
        plt.show()

        return img_transformed # torch.Size()
    
