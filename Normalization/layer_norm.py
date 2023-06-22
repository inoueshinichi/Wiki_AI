"""レイヤー正規化
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

# Layer Normalization
class LayerNormApp:
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

        # Input : (N, C, H, W), (N, C, D), etc = (N, *)

        v_t = torch.Tensor([[1, 2, 3], [4, 5, 6]])
        print('v_t.shape', v_t.shape)
        print('v_t: \n', v_t)
        print('v_t.mean((-2,-1))', v_t.mean((-2,-1)))

        N = 2
        C = 3
        H = 2
        W = 2
        input = torch.ones(N,C,H,W)
        print('input.shape', input.shape)
        print('input: \n', input)

        output = input.mean((-2, -1))
        print('output.shape', output.shape)
        print('output: \n', output)

        # 次元(H=2,W=2)の4つの要素からスカラの平均値と標準偏差を計算して,
        # (H=2,W=2)のテンソルに一様にAffine変換を行う(if elementwise_affine=True).
        # Affine変換時のスケールパラメータγとシフトパラメータβの形状は(H=2,W=2).
        layer_norm = nn.LayerNorm(normalized_shape=(H,W), elementwise_affine=True)
        
        gamma = layer_norm.weight # γ (スケールパラメータ with leanable)
        beta = layer_norm.bias # β (シフトパラメータ with learnable)
        print('gamma.shape', gamma.shape)
        print('gamma: \n', gamma)
        print('beta.shape',beta.shape)
        print('beta: \n', beta)

        output_layer_norm = layer_norm(input)
        print('output_layer_nrom.shape', output_layer_norm.shape)
        print('output_layer_nrom: \n', output_layer_norm)


