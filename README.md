# Wiki_AI_Summary

## My survey about DL basic component and various task models

### Pytorchメモ
+ v2.0.0 Python API https://pytorch.org/docs/stable/index.html
+ メモ https://github.com/inoueshinichi/Wiki_AI/wiki/Wiki_pytorch_memo
+ Dataset https://github.com/inoueshinichi/Wiki_AI/wiki/Wiki_pytorch_dataset
+ Dataloader https://github.com/inoueshinichi/Wiki_AI/wiki/Wiki_pytorch_datasetloader
+ Module https://github.com/inoueshinichi/Wiki_AI/wiki/Wiki_pytorch_module
+ Device(cpu, cuda-gpu) https://github.com/inoueshinichi/Wiki_AI/wiki/Wiki_pytorch_cpu_gpu
+ Tensorboard https://github.com/inoueshinichi/Wiki_AI/wiki/Wiki_pytorch_tensorboard
+ Deploy (ONNX, TorchScript, Trace, C++ libtorch) https://github.com/inoueshinichi/Wiki_AI/wiki/Wiki_pytorch_deploy

### Tensorflowメモ
+ v2.13.0 Python API https://www.tensorflow.org/api_docs/python/tf
+ Tensorflow Liteは, 基本的に同じバージョンを使用すること

### Tensorflow Liteメモ
+ FlatBuffer(*.tflite) https://flatbuffers.dev
+ tflite-runtime v2.13.0 https://pypi.org/project/tflite-runtime/
+ tflite-support v0.4.4 https://github.com/tensorflow/tflite-support
+ 推論方法 https://www.tensorflow.org/lite/guide/inference?hl=ja

### Tensorflow Liteの使い方
+ https://github.com/inoueshinichi/Wiki_AI/wiki/Wiki_TensorflowLiteの使い方

### フレームワーク間のモデル変換
+ https://github.com/inoueshinichi/Wiki_AI/wiki/Wiki_Framework_Model_Change

### 関数の微分と連鎖律
+ https://github.com/inoueshinichi/Wiki_AI/wiki/Wiki_Defferencial_ChainRule

### 学習形態
＋ https://github.com/inoueshinichi/Wiki_AI/wiki/Wiki_Training_Style

### データセット
+ https://github.com/inoueshinichi/Wiki_AI/wiki/Wiki_Dataset

### DL基本コンポーネント
| Name | URL |
| :-- | :-- |
| データ拡張 | https://github.com/inoueshinichi/Wiki_AI/wiki/Wiki_Data_Augmentation |
| 活性化関数 | https://github.com/inoueshinichi/Wiki_AI/wiki/Wiki_Activation |
| 損失関数 | https://github.com/inoueshinichi/Wiki_AI/wiki/Wiki_Loss |
| 最適化方策 | https://github.com/inoueshinichi/Wiki_AI/wiki/Wiki_Optimizer | 
| 標準化 | https://github.com/inoueshinichi/Wiki_AI/wiki/Wiki_Regularization |
| 正規化 | https://github.com/inoueshinichi/Wiki_AI/wiki/Wiki_Normalization |
| FCN(Linear) | https://github.com/inoueshinichi/Wiki_AI/wiki/Wiki_Linear |
| 注意機構 | https://github.com/inoueshinichi/Wiki_AI/wiki/Wiki_Attention |
| 畳み込み |https://github.com/inoueshinichi/Wiki_AI/wiki/Wiki_Convolution | 
| プーリング | https://github.com/inoueshinichi/Wiki_AI/wiki/Wiki_Pooling |
| スキップ接続 | https://github.com/inoueshinichi/Wiki_AI/wiki/Wiki_Skip_Connection |
| Embedding | https://github.com/inoueshinichi/Wiki_AI/wiki/Wiki_Embedding | 
| Transformer | https://github.com/inoueshinichi/Wiki_AI/wiki/Wiki_Transformer |

### 学習時のクラスバランス調整
| Name | URL |
| :-- | :-- |
| ミニバッチ内ラベルの均衡化 | https://github.com/inoueshinichi/Wiki_AI/wiki/Wiki_Balance_Label |
| 重み付き損失 | https://github.com/inoueshinichi/Wiki_AI/wiki/Wiki_Weight_Loss　|

### 正則化(過学習を抑制)
| Name | URL |
| :-- | :-- |
| ドロップアウト | https://github.com/inoueshinichi/Wiki_AI/wiki/Wiki_Dropout |
| L1/L2ノルム(Weight Decay) | |
| ラベル平均化 | |

### 特徴量抽出機構と識別機構(Backborn & Head)
| Name | URL |
| :-- | :-- |
| Backboarn | https://github.com/inoueshinichi/Wiki_AI/wiki/Wiki_Backborn_Models |
| Head | https://github.com/inoueshinichi/Wiki_AI/wiki/Wiki_Head_Models |

### タスク別モデル
| Name | URL |
| :-- | :-- |
| Image | https://github.com/inoueshinichi/Wiki_AI/wiki/Wiki_Task_Image |
| NPL | https://github.com/inoueshinichi/Wiki_AI/wiki/Wiki_Task_NPL |
| V&L | https://github.com/inoueshinichi/Wiki_AI/wiki/Wiki_Task_V&L |
| GAN | https://github.com/inoueshinichi/Wiki_AI/wiki/Wiki_Task_GAN |
| Video | https://github.com/inoueshinichi/Wiki_AI/wiki/Wiki_Task_Video |
| 3D | https://github.com/inoueshinichi/Wiki_AI/wiki/Wiki_Task_3D |



### 開発ツール
+ Pytorch https://pytorch.org
+ Tensorflow https://www.tensorflow.org/?hl=ja
+ Tensorflow-Lite https://www.tensorflow.org/lite?hl=ja
+ Neural Network Console https://dl.sony.com/ja/

### 管理ツール
+ TensorBoard https://www.tensorflow.org/tensorboard?hl=ja
+ Netron https://netron.app
+ MLFlow https://mlflow.org
+ Comet https://www.comet.com/site/
