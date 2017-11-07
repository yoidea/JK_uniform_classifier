from keras.models import Sequential
from keras.layers import Activation, Dense, Dropout
from keras.layers import Conv2D, MaxPooling2D, Flatten

    
def build_model(input_size, output_size):
    """
    モデルを構築
    # 引数
        input_dim : Integer, 入力ノイズの次元
        output_size : List, 出力画像サイズ
    # 戻り値
        model : Keras model, 生成器
    """
    model = Sequential()
    input_shape = (input_size[0], input_size[1], 3)
    model.add(Conv2D(16, kernel_size=(3, 3), input_shape=input_shape))
    model.add(Activation('relu'))
    model.add(Dropout(0.75))
    model.add(Conv2D(32, kernel_size=(3, 3)))
    model.add(Activation('relu'))
    model.add(Dropout(0.75))
    model.add(Conv2D(64, kernel_size=(3, 3)))
    model.add(Activation('relu'))
    model.add(Dropout(0.75))
    model.add(Conv2D(128, kernel_size=(3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.75))
    model.add(Flatten())
    model.add(Dense(256))
    model.add(Activation('relu'))
    model.add(Dropout(0.75))
    model.add(Dense(output_size))
    model.add(Activation('softmax'))
    return model