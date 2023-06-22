"""バッチ正規化
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

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from type_hint import *


# Batch Normalization 1D
class BatchNorm1dApp:
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

        # Must Input : (N, C) or (N, C, L)
        N = 2
        C = 3
        L = 4

        print('(N, C) input [L=1]--------------------------------------')

        # (N, C)
        input_2dim = torch.ones(N, C)
        print('input_2dim.shape', input_2dim.shape)
        print('input_2dim: \n', input_2dim)

        batch_norm_1d = nn.BatchNorm1d(num_features=C) # 平均値と標準偏差をベクトルとして集約したい次元を指定
        
        gamma = batch_norm_1d.weight
        beta = batch_norm_1d.bias
        print('gamma.shape', gamma.shape)
        print('gamma: \n', gamma)
        print('beta.shape', beta.shape)
        print('beta: \n', beta)

        output_2dim = batch_norm_1d(input_2dim)
        print('output_2dim.shape', output_2dim.shape)
        print('output_2dim: \n', output_2dim)

        print('(N, C, L) input [Main]--------------------------------------')

        # (N, C, L)
        input_3dim = torch.ones(N, C, L)
        print('input_3dim.shape', input_3dim.shape)
        print('input_3dim: \n', input_3dim)

        batch_norm_1d = nn.BatchNorm1d(num_features=C) # 平均値と標準偏差をベクトルとして集約したい次元を指定
        
        gamma = batch_norm_1d.weight
        beta = batch_norm_1d.bias
        print('gamma.shape', gamma.shape)
        print('gamma: \n', gamma)
        print('beta.shape', beta.shape)
        print('beta: \n', beta)

        output_3dim = batch_norm_1d(input_3dim)
        print('output_3dim.shape', output_3dim.shape)
        print('output_3dim: \n', output_3dim)


# Batch Normalization 2D
class BatchNorm2dApp:
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

        # Must Input : (N, C, H, W)
        N = 2
        C = 3
        H = 4
        W = 4

        batch_norm_2d = nn.BatchNorm2d(num_features=C) # 平均値と標準偏差をベクトルとして集約したい次元を指定

        print('(N, C, H, W) input [Main]--------------------------------------')

        input_4dim = torch.ones(N, C, H, W)
        print('input_4dim.shape', input_4dim.shape)
        print('input_4dim: \n', input_4dim)

        output_4dim = batch_norm_2d(input_4dim)

        gamma = batch_norm_2d.weight
        beta = batch_norm_2d.bias
        print('gamma.shape', gamma.shape)
        print('gamma: \n', gamma)
        print('beta.shape', beta.shape)
        print('beta: \n', beta)

        print('output_4dim.shape', output_4dim.shape)
        print('output_4dim: \n', output_4dim)


# Batch Normalization 3D
class BatchNorm3dApp:
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

        # Must Input : (N, C, D, H, W)
        N = 2
        C = 3
        D = 4
        H = 4
        W = 4

        batch_norm_3d = nn.BatchNorm3d(num_features=C) # 平均値と標準偏差をベクトルとして集約したい次元を指定

        print('(N, C, D, H, W) input [Main]--------------------------------------')

        input_5dim = torch.ones(N, C, D, H, W)
        print('input_5dim.shape', input_5dim.shape)
        print('input_5dim: \n', input_5dim)


        gamma = batch_norm_3d.weight
        beta = batch_norm_3d.bias
        print('gamma.shape', gamma.shape)
        print('gamma: \n', gamma)
        print('beta.shape', beta.shape)
        print('beta: \n', beta)

        output_5dim = batch_norm_3d(input_5dim)
        print('output_5dim.shape', output_5dim.shape)
        print('output_5dim: \n', output_5dim)


