"""学習済みモデルによる推論(画像分類)
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
from .preprocess import BaseTransform


# ILSVRCのラベルを用いた推論クラス
class ILSVRCPredictor():

    def __init__(self, class_index):
        self.class_index = class_index

    def predict_max(self, out):
        maxID = np.argmax(out.detach().numpy())
        predicted_label_name = self.class_index[str(maxID)][1]
        return predicted_label_name
    

class PretrainedModelCheckApp:
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

        # VGG-16モデルのインスタンスを生成
        use_pretained = True
        net = models.vgg16(pretrained=use_pretained)
        net.eval()

        # モデルのネットワーク構成を出力
        print('pretrained model: ', net)


class PretrainedModelInferenceApp:
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

        # VGG-16モデルのインスタンスを生成
        use_pretained = True
        net = models.vgg16(pretrained=use_pretained)
        net.eval()

        # モデルのネットワーク構成を出力
        # print('pretrained model: ', net)

        # 推論ラベルの準備(ILSVRC)
        ILSVRC_class_index = json.load(open(f'{os.path.dirname(__file__)}/data/imagenet_class_index.json', 'r'))
        print(ILSVRC_class_index)
        print("ILSVRC_class_index :")
        for index_str in ILSVRC_class_index:
            print(f"Index[{index_str}], Label[{ILSVRC_class_index[index_str]}]")

        # ILSVRCPredictor
        predictor = ILSVRCPredictor(ILSVRC_class_index)

        # 画像読み込み
        image_file_path = f"{os.path.dirname(__file__)}/data/goldenretriever-3724972_640.jpg"
        img = Image.open(image_file_path)

        # 前処理
        resize = 224
        mean = (0.485, 0.456, 0.406)
        std = (0.229, 0.224, 0.225)
        transform = BaseTransform(resize, mean, std)
        img_transformed = transform(img) # torch.Size([3, 224, 224])

        # バッチサイズの次元を追加する
        inputs_t = img_transformed.unsqueeze_(0) # torch.Size([1, 3, 224, 224])

        # 推論
        out = net(inputs_t) # torch.Size([1, 1000])
        result = predictor.predict_max(out)

        # 予測結果を出力する
        print("入力画像の予測結果 :", result)
