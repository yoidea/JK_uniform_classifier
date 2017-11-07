import os
import numpy as np
from PIL import Image
from tqdm import tqdm
from keras.utils.np_utils import to_categorical


def to_dirname(name):
    """
    ディレクトリ名の"/"有無の違いを吸収する
    # 引数
        name : String, ディレクトリ名
    # 戻り値
        name : String, 変更後
    """
    if name[-1:] == '/':
        return name
    else:
        return name + '/'


def load_data(name, size, ext='.jpg'):
    """
    画像群とラベルを対応させ教師データを作成する
    # 引数
        name : String, 保存場所
        size : List, 画像サイズ
        ext : String, 拡張子
    # 戻り値
        sets : Numpy array, 教師データ
    """
    i = 0
    names = []
    labels = np.array([])
    sets = np.empty((0, size[0], size[1], 3))
    for dirname in os.listdir(name):
        if os.path.isdir(name+dirname):
            names.append(dirname)
            images = load_images(name+dirname+'/', size, ext)
            sets = np.concatenate((sets, images))
            labels = np.append(labels, [i] * len(images))
            i = i + 1
    sets = np.array(sets)
    labels = to_categorical(np.array(labels))
    return (sets, labels, names)


def load_image(name, size):
    """
    画像を読み込み配列に格納する
    # 引数
        name : String, 保存場所
        size : List, 画像サイズ
    # 戻り値
        image : Numpy array, 画像データ
    """
    image = Image.open(name)
    image = image.resize(size)
    image = np.array(image)
    image = image / 255
    # モデルの入力次元にあわせる
    image = np.array([image])
    return image


def load_images(name, size, ext='.jpg'):
    """
    画像群を読み込み配列に格納する
    # 引数
        name : String, 保存場所
        size : List, 画像サイズ
        ext : String, 拡張子
    # 戻り値
        images : Numpy array, 画像データ
    """
    images = np.empty((0, size[0], size[1], 3))
    for file in tqdm(os.listdir(name)):
        if os.path.splitext(file)[1] != ext:
            # 拡張子が違うなら処理しない
            continue
        image = Image.open(name+file)
        if image.mode != "RGB":
            # 3ch 画像でなければ変換する
            image.convert("RGB")
        image = image.resize(size)
        image = np.array(image)
        images = np.concatenate((images, [image]))
    # 256階調のデータを0-1の範囲に正規化する
    images = images / 255
    return images