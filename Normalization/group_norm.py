"""グループ正規化
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


class GroupNormApp:
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

        # Must Input : (N, C, *)
        N = 2
        C = 9
        H = 2
        W = 2

        print('(N, C, H, W) input ------------------------------------')

        input_4dim = torch.ones(N, C, H, W)
        print('input_4dim.shape', input_4dim.shape)
        print('input_4dim: \n', input_4dim)

        groups = 3
        channels = C
        affine = True
        # 次元C=9を3分割して以降の次元をまとめた上で統計量を取る. 
        # スケールパラメータγとシフトパラメータβは, C/3 = 3次元ベクトルになるが, C=9に拡張される.
        group_norm = nn.GroupNorm(num_groups=groups, num_channels=channels, affine=affine)

        if affine:
            gamma = group_norm.weight
            beta = group_norm.bias
            print('gamma.shape', gamma.shape)
            print('gamma: \n', gamma)
            print('beta.shape', beta.shape)
            print('beta: \n', beta)

