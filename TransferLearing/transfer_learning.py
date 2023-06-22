"""アリとハチを分類するタスクに対する転移学習
"""
import os
import sys

from traitlets import default

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
import shutil
import hashlib
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
from torch.utils.data import DataLoader
#torch.backends.cudnn.deterministic = True
#torch.backends.cudnn.benchmark = False
from torch.utils.tensorboard import SummaryWriter

from type_hint import *
from .preprocess import ImageTransform
from .ant_bee_dataset import HymenopteraDataset, make_hymenoptera_dataset
from enum_with_estimate import enumerate_with_estimate

METRICS_LABEL_NDX = 0 # アリ:1, ハチ:2
METRICS_POS_PRED_NDX = 1 # アリの尤度
METRICS_NEG_PRED_NDX = 2 # ハチの尤度
METRICS_FP_LOSS_NDX = 3 # ラベル: 陰性, 推論: 陽性 (過検出)
METRICS_FN_LOSS_NDX = 4 # ラベル: 陽性, 推論: 陰性 (見逃し)
METRICS_ALL_LOSS_NDX = 5 # 全損失
METRICS_FP_NDX = 6 # 偽陽性
METRICS_TP_NDX = 7 # 真陽性
METRICS_FN_NDX = 8 # 偽陰性
METRICS_TN_NDX = 9 # 真陰性
METRICS_SIZE = 10 # 配列の行数

class TransferLearningApp:
    def __init__(self, sys_argv : Optional[Any] = None):
        if sys_argv is None:
            sys_argv = sys.argv[1:]

        parser : Any = argparse.ArgumentParser()

        # 必要であれば, ここにアプリ引数を登録
        parser.add_argument('--batch-size',
                            help='Batch size to use for training.',
                            default=24,
                            type=int,
                            )
        parser.add_argument('--num-workers',
                            help='Number of worker processes for background data loading.',
                            default=4,
                            type=int,
                            )
        parser.add_argument('--epochs',
                            help='Number of epochs to train for',
                            default=1,
                            type=int,
                            )
        parser.add_argument('--threshold',
                            help='Classification threshold.',
                            default=0.5,
                            type=float,
                            )
        parser.add_argument('--seed',
                            help='Random seed',
                            default=1234,
                            type=int,
                            )
        
        # Tensorboard
        app_signigure = "hymenoptera"
        parser.add_argument('--tb-prefix',
                            default=app_signigure,
                            help='Data prefix to use for Tensorboard run.',
                            )
        tb_logdir = "runs"
        parser.add_argument("--tb-logdir",
                            default=tb_logdir,
                            help='Log directory of Tensorbard.',
                            )
        parser.add_argument('comment',
            help='Comment suffix for Tensorboard run',
            nargs='?',
            default='dwlpt',
        )

        # Save Model Directory
        save_modeldir = "models"
        parser.add_argument('--save-modeldir',
                            default=save_modeldir,
                            help='Save directory of model state.',
                            )
        
        # Option for FineTuning
        parser.add_argument('--finetune',
                            help='Start finetuning from this model.',
                            default='',
                            )
        parser.add_argument('--finetune-depth',
                            help='Number of blocks (counted from the headd) to include in finetuning.',
                            type=int,
                            default=1,
                            )

        self.cli_args = parser.parse_args(sys_argv)
        self.time_str = datetime.datetime.now().strftime('%Y-%m-%d_%H.%M.%S')

        self.use_cuda = torch.cuda.is_available()
        self.device = torch.device('cuda' if self.use_cuda else 'cpu')

        self.classes_dict = {
            'ants' : 1,
            'bees' : 2,
        }

        # 乱数シード (共通)
        torch.manual_seed(self.cli_args.seed)
        np.random.seed(self.cli_args.seed)
        random.seed(self.cli_args.seed)

        self.trn_writer = None
        self.val_writer = None
        self.total_training_samples_count : int = 0
        self.model = self.init_model()
        self.optimizer = self.init_optim()
        self.trn_ds, self.val_ds = self.init_dataset()

    def init_model(self) -> Any:
        # 学習済みモデルVGG-16をロード
        model = models.vgg16(pretrained=True)

        # 出力層の出力ユニットをアリとハチの2つに付け替える
        model.classifier[6] = nn.Linear(in_features=4096, out_features=2)

        if self.cli_args.finetune:
            model_blocks = [
                n for n, subm in model.named_children()
                if len(list(subm.parameters())) > 0
            ]

            finetune_blocks = model_blocks[-self.cli_args.finetune_depth:]
            log.info(f"finetuning from {self.cli_args.finetune}, blocks {' '.join(finetune_blocks)}")

            for n, p in model.named_parameters():
                if n.split('.')[0] not in finetune_blocks:
                    p.requires_grad_(False) # 勾配計算をさせない (凍結)

        if self.use_cuda:
            log.info('Using CUDA; {} device.'.format(torch.cuda.device_count()))
            if torch.cuda.device_count() > 1:
                model = nn.DataParallel(model)
            model = model.to(self.device)

        return model

    def init_optim(self) -> Any:
        lr = 0.003 if self.cli_args.finetune else 0.001
        # Momentum SGD
        return optim.SGD(params=self.model.parameters(), 
                         lr=lr, 
                         momentum=0.9, 
                         weight_decay=1e-4,
                         )

    def init_dataset(self) -> Tuple[HymenopteraDataset, HymenopteraDataset]:
        return make_hymenoptera_dataset()
    
    def init_trn_dl(self) -> DataLoader:
        batch_size = self.cli_args.batch_size
        if self.use_cuda:
            batch_size *= torch.cuda.device_count()

        trn_dl = DataLoader(
            dataset=self.trn_ds,
            batch_size=self.cli_args.batch_size,
            num_workers=self.cli_args.num_workers,
            pin_memory=self.use_cuda,
            shuffle=True, # 訓練データセットはシャッフルする
            )
        
        return trn_dl
    
    def init_val_dl(self) -> DataLoader:
        batch_size = self.cli_args.batch_size
        if self.use_cuda:
            batch_size *= torch.cuda.device_count()
        val_dl = DataLoader(
            dataset=self.val_ds,
            batch_size=self.cli_args.batch_size,
            num_workers=self.cli_args.num_workers,
            pin_memory=self.use_cuda,
        )
        return val_dl
    
    def init_tensorboard_writer(self):
        if self.trn_writer is None:
            log_dir = os.path.join('runs', self.cli_args.tb_prefix, self.time_str)
        
            self.trn_writer = SummaryWriter(
                log_dir=log_dir + '-trn_cls-' + self.cli_args.comment
            )
            self.val_writer = SummaryWriter(
                log_dir=log_dir + '-val_cls-' + self.cli_args.comment
            )

    def do_training(self,
                    epoch_ndx : int, 
                    trn_dl : DataLoader,
                    ):
        # トレーニングモード
        self.model.train()

        # 訓練データの計測データ用配列データ
        trn_metrics_g = torch.zeros(
            (METRICS_SIZE, len(trn_dl.dataset)),
            device=self.device,
        )

        batch_iter = enumerate_with_estimate(
            trn_dl,
            "E{} Training".format(epoch_ndx),
            start_ndx=trn_dl.num_workers,
        )

        # batch_tup [img_t, label_t]
        for batch_ndx, batch_tup in batch_iter:

            self.optimizer.zero_grad()

            loss_var = self.compute_batch_loss(
                batch_ndx,
                batch_tup,
                trn_dl.batch_size,
                trn_metrics_g,
            )

            # Back Propagation
            loss_var.backward()

            # Update weight parameters
            self.optimizer.step()

        self.total_training_samples_count += len(trn_dl.dataset)

        return trn_metrics_g.to('cpu')

    def compute_batch_loss(self, 
                           batch_ndx, 
                           batch_tup, 
                           batch_size, 
                           metrics_g):
        
        input_t, label_t = batch_tup

        input_g = input_t.to(self.device, non_blocking=True)
        label_g = label_t.to(self.device, non_blocking=True)

        logits_g = self.model(input_g) # (score, socre)

        print('----- check -----')
        print('input_g.size', input_g.size())
        print('label_g.size', label_g.size())

        # reduction='none'でサンプル毎の損失を計算
        loss_func = nn.CrossEntropyLoss(reduction='none')
        loss_g = loss_func(
            logits_g, 
            label_g,
        )

        probability_g = nn.Softmax(dim=-1)(logits_g)
        print('probability_g.size', probability_g.size())

        start_ndx = batch_ndx * batch_size
        end_ndx = start_ndx + label_t.size(0) # ミニバッチ数(端数あり)

        """勾配を必要とする指標がないのでデタッチして, 計算グラフから切り離す."""
        # Label
        metrics_g[METRICS_LABEL_NDX, start_ndx:end_ndx] = np.where(label_g[:,0].detach() == 0, 
                                                                   self.classes_dict['ants'], 
                                                                   self.classes_dict['bees']) # ants: 1, bees:2
        # Predict
        metrics_g[METRICS_POS_PRED_NDX, start_ndx:end_ndx] = probability_g[:,0].detach()
        metrics_g[METRICS_NEG_PRED_NDX, start_ndx:end_ndx] = probability_g[:,1].detach()

        # Mask
        pos_label_mask = metrics_g[METRICS_LABEL_NDX] == self.classes_dict['ants']
        pos_pred_mask = metrics_g[METRICS_POS_PRED_NDX] > self.cli_args.threshold # predict ants
        neg_pred_mask = ~pos_pred_mask # predict bees
        neg_label_mask = ~pos_label_mask #  bees

        # Metrics
        metrics_g[METRICS_TP_NDX, start_ndx:end_ndx] = pos_label_mask & pos_pred_mask
        metrics_g[METRICS_FN_NDX, start_ndx:end_ndx] = pos_label_mask & ~pos_pred_mask
        metrics_g[METRICS_TN_NDX, start_ndx:end_ndx] = neg_label_mask & neg_pred_mask
        metrics_g[METRICS_FN_NDX, start_ndx:end_ndx] = neg_label_mask & ~neg_pred_mask
        
        # Loss
        metrics_g[METRICS_ALL_LOSS_NDX, start_ndx:end_ndx] = loss_g.detach()

        return loss_g.mean() # サンプル毎の損失を1バッチ分に平均化
    
    def do_validation(self,
                      epoch_ndx : int, 
                      val_dl : DataLoader,
                      ):
        with torch.no_grad():
            
            val_metrics_g = torch.zeros(
                (METRICS_SIZE, len(val_dl.dataset)),
                device=self.device,
                )
            
            # 検証モード
            self.model.eval()

            batch_iter = enumerate_with_estimate(
                            val_dl,
                            "E{} Validation".format(epoch_ndx),
                            start_ndx=val_dl.num_workers,
                            )
            
            # batch_tup [img_t, label_t]
            for batch_ndx, batch_tup in batch_iter:
                self.compute_batch_loss(
                    batch_ndx,
                    batch_tup,
                    val_dl.batch_size,
                    val_metrics_g,
                )
            
        return val_metrics_g.to('cpu')

    def main(self):
        """アプリケーション
        """
        log.info("Starting {}, {}".format(type(self).__name__, self.cli_args))


        trn_dl = self.init_trn_dl()
        val_dl = self.init_val_dl()

        best_score = 0.0
        validation_cadence = 5 if not self.cli_args.finetune else 1

        for epoch_ndx in range(1, self.cli_args.epochs + 1):

            log.info("Epoch {} of {}, [trn]{}/[val]{} batches of size {}*{}".format(
                epoch_ndx,
                self.cli_args.epochs,
                len(trn_dl),
                len(val_dl),
                self.cli_args.batch_size,
                (torch.cuda.device_count() if self.use_cuda else 1),
            ))

            trn_metrics_t = self.do_training(epoch_ndx, trn_dl)
            self.log_metrics(epoch_ndx, 'trn', trn_metrics_t)

            if epoch_ndx == 1 or epoch_ndx % validation_cadence == 0:
                valMetrics_t = self.do_validation(epoch_ndx, val_dl)
                score = self.log_metrics(epoch_ndx, 'val', valMetrics_t)
                best_score = max(score, best_score)

                # TODO: this 'cls' will need to change for the malignant classifier
                self.save_model('cls', epoch_ndx, score == best_score)

        if hasattr(self, 'trn_writer'):
            self.trn_writer.close()
            self.val_writer.close()

    
    def log_metrics(self, 
                    epoch_ndx : int, 
                    mode_str : str,
                    metrics_t : torch.Tensor,
                    ):
        log.info("E{} {}".format(
            epoch_ndx,
            type(self).__name__,
        ))

        metrics_a = metrics_t.detach().numpy()
        sum_a = metrics_a.sum(axis=1) # 行方向に加算
        assert np.isfinite(metrics_a).all()

        all_label_count = sum_a[METRICS_TP_NDX] + sum_a[METRICS_FN_NDX]

        metrics_dict = {}
        metrics_dict['loss/all'] = metrics_a[METRICS_ALL_LOSS_NDX].mean()
        metrics_dict['percent_all/tp'] = sum_a[METRICS_TP_NDX] / (all_label_count or 1) * 100
        metrics_dict['percent_all/fn'] = sum_a[METRICS_FN_NDX] / (all_label_count or 1) * 100
        metrics_dict['percent_all/fp'] = sum_a[METRICS_FP_NDX] / (all_label_count or 1) * 100
        metrics_dict['percent_all/tn'] = sum_a[METRICS_TN_NDX] / (all_label_count or 1) * 100

        precision = metrics_dict['pr/precision'] = sum_a[METRICS_TP_NDX] / ((sum_a[METRICS_TP_NDX] + sum_a[METRICS_FP_NDX]) or 1)
        recall = metrics_dict['pr/recall'] = sum_a[METRICS_TP_NDX] / ((sum_a[METRICS_TP_NDX] + sum_a[METRICS_FN_NDX]) or 1)

        metrics_dict['pr/f1_score'] = 2 * (precision * recall) /((precision + recall) or 1)

        log.info(("E{} {:8} "
                  + "{loss/all:.4f} loss, "
                  + "{pr/precision:.4f} precision, "
                  + "{pr/recall:.4f} recall, "
                  + "{pr/f1_score:.4f} f1 score"
                  ).format(
            epoch_ndx,
            mode_str,
            **metrics_dict,
        ))

        log.info(("E{} {:8} "
                  + "{loss/all:.4f} loss, "
                  + "{percent_all/tp:-5.1f}% tp, {percent_all/fn:-5.1f}% fn, {percent_all/fp:-9.1f}% fp"
                  ).format(
            epoch_ndx,
            mode_str + '_all',
            **metrics_dict,
        ))

        self.init_tensorboard_writer()
        writer = getattr(self, mode_str + '_writer')

        # ここにTensorboardに出力したい計測値を追加する

        score = metrics_dict['pr/recall']

        return score

    def save_model(self,
                   type_str : str,
                   epoch_ndx : int,
                   is_best : bool = False,
                    ):
    
        file_path = os.path.join(
            f"{os.path.dirname(__file__)}",
            self.cli_args.save_modeldir,
            self.cli_args.tb_prefix,
            '{}_{}_{}.{}.state'.format(
                type_str,
                self.time_str,
                self.cli_args.comment,
                self.total_training_samples_count,
            )
        )

        os.makedirs(os.path.dirname(file_path), mode=0o755, exist_ok=True)

        model = self.model
        if isinstance(model, torch.nn.DataParallel):
            model = model.module # DataParallelでラップされている場合は, 取り出す.

        state = {
            'sys_argv': sys.argv,
            'time': str(datetime.datetime.now()),
            'model_state': model.state_dict(), # モデルパラメータ
            'model_name': type(model).__name__,
            'optimizer_state': self.optimizer.state_dict(), # オプティマイザパラメータ
            'optimizer_name': type(self.optimizer).__name__,
            'epoch': epoch_ndx,
            'total_training_samples_count': self.total_training_samples_count,
        }

        torch.save(state, file_path)

        log.info("Saved model params to {}".format(file_path))

        if is_best:
            best_path = os.path.join(
                f"{os.path.dirname(__file__)}",
                self.cli_args.save_modeldir,
                self.cli_args.tb_prefix,
                f'{type_str}_{self.time_str}_{self.cli_args.comment}.best.state'
            )
            shutil.copyfile(file_path, best_path)

            log.info("Saved model params to {}".format(best_path))

        with open(file_path, 'rb') as f:
            log.info("SHA1: " + hashlib.sha1(f.read()).hexdigest())


