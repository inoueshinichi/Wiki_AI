"""データセットのダウンロードと解凍
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

import urllib
import urllib.request
import zipfile

def main():
    # 1.3節で使用するアリとハチの画像データをダウンロードし解凍します
    # PyTorchのチュートリアルで用意されているものです
    # https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html

    url = "https://download.pytorch.org/tutorial/hymenoptera_data.zip"
    data_dir = f"{os.path.dirname(__file__)}/data"
    save_path = os.path.join(data_dir, "hymenoptera_data.zip")

    if not os.path.exists(save_path):
        urllib.request.urlretrieve(url, save_path)

        # ZIPファイルを読み込み
        zip = zipfile.ZipFile(save_path)
        zip.extractall(data_dir)  # ZIPを解凍
        zip.close()  # ZIPファイルをクローズ

        # ZIPファイルを消去
        os.remove(save_path)

if __name__ == "__main__":
    main()