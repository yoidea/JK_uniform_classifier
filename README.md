# JK_uniform_classifier

Predicting school name from images of JK uniform

## 概要

JKの制服画像を読み込んで学校名を推定できる装置。
理論上はデータセットが集まればJK以外の制服にも適用可能。

## 依存関係

- Python 3.0 or more
- Keras 2.0 or more (Tensorflow backend)
- Pillow
- numpy
- tqdm
- h5py

## 使用法

0. リポジトリをクローンする。
```
git clone https://github.com/yoidea/JK_uniform_classifier.git
```

1. まず、JKの制服画像を収集する。くれぐれも違法行為を行わないように。
データセット用のディレクトリを作成して、その中に1校につき1ディレクトリを作成して、画像を格納する。
(例) 3校の画像を20枚ずつ集めた場合
```
ls images/
南高校     北高校     西高校     東高校
ls images/北高校/
01.jpg     02.jpg     ...     20.jpg
```

2. 引数にデータセット用のディレクトリを指定して`train.py`を実行し、学習を開始する。
```
python train.py --input images/
```

3. `classify.py`を実行して、判定したい画像のパスを入力すると判定が行われる。
```
python classify.py
```

## 実行例
地元の3高校の制服画像60枚(1校20枚ずつ)を学習し、別に用意したテスト用の画像でテストを行った。(正解 : ●●北高校)
```
python classify.py
Using TensorFlow backend.
>> test/kita.jpg
●●北高校
 95% ############################

北●●高校
  2% 

●●台高校
  1% 

```

精度が悪いこともある。(正解 : ●●台高校)
```
python classify.py
Using TensorFlow backend.
>> test/nodai.jpg
●●北高校
 15% ####

北●●高校
 27% ########

●●台高校
 57% #################

```
