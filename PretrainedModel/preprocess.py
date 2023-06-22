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

from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision import models, transforms

from type_hint import *


# 入力画像の前処理クラス
# 224 x 224 リサイズ
# 平均(0.485, 0.456, 0.406), 標準偏差(0.229, 0.224, 0.225)　規格化
class BaseTransform():
    def __init__(self, resize, mean, std):
        self.base_transform = transforms.Compose([
            transforms.Resize(resize),      # 画像の短辺が244になるようにリサイズ
            transforms.CenterCrop(resize),  # 画像中央を切り抜く(resize x resize)
            transforms.ToTensor(),          # Torchテンソルに変換
            transforms.Normalize(mean, std) # 色情報の規格化
        ])

    def __call__(self, img):
        return self.base_transform(img)
    


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
        transform = BaseTransform(resize, mean, std)
        img_transformed = transform(img) # torch.Size([3, 224, 224])
        img_transformed_transposed = img_transformed.numpy().transpose((1,2,0))

        print("img_transformed_transposed :",  img_transformed_transposed)
        img_transformed_transposed_cliped = np.clip(img_transformed_transposed, 0, 1) # 0-1にクリップ
        print("img_transformed_transposed_cliped :",  img_transformed_transposed_cliped)
        plt.imshow(img_transformed_transposed_cliped)
        plt.show()

        return img_transformed # torch.Size([3, 224, 224])