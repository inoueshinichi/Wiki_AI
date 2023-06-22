"""画像の前処理
"""
import os
import sys

# os.sepはプラットフォーム固有の区切り文字(Windows: `\`, Unix: `/`)
module_parent_dir = os.sep.join([os.path.dirname(__file__), '..'])
print("module_parent_dir", module_parent_dir)
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
from .conf_random_seed import *

# 乱数シード (共通)
torch.manual_seed(1234)
np.random.seed(1234)
random.seed(1234)


# 前処理クラス
class ImageTransform():

    def __init__(self, resize, mean, std):
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

        
