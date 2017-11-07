import os
from json import dumps, loads
from keras.models import model_from_json


def save_list(data, name):
    """
    ListをJSONに変換し、任意の名前で保存する
    # 引数
        data : List
        name : String, 保存先ファイル名
    """
    json = dumps(data)
    with open(name, 'w') as f:
        f.write(json)


def load_list(name):
    """
    任意のJSONファイルを参照し、Listに変換する
    # 引数
        name : String, 保存先ファイル名
    # 戻り値
        data : List
    """
    with open(name) as f:
        json = f.read()
    data = loads(json)
    return data


def save_model(model, name):
    """
    モデルをJSONに変換し、任意の名前で保存する
    # 引数
        model : Keras model
        name : String, 保存先ファイル名
    """
    json = model.to_json()
    with open(name, 'w') as f:
        f.write(json)


def load_model(name):
    """
    任意のJSONファイルを参照し、モデルに変換する
    # 引数
        name : String, 保存先ファイル名
    # 戻り値
        Keras model
    """
    with open(name) as f:
        json = f.read()
    model = model_from_json(json)
    return model