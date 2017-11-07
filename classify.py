import os
import numpy as np
from utils.image import load_image
from utils.file import load_model, load_list


def display(names, prediction):
    for i in range(len(names)):
        print(names[i])
        print('{0:3d}'.format(int(prediction[0][i] * 100)), end='% ')
        for j in range(int(prediction[0][i] * 30)):
            print('#', end='')
        print('\n')


def main():
    model = load_model('model.json')
    model.load_weights('weights.hdf5')
    names = load_list('names.json')
    print('Enter the file name (*.jpg)')
    while True:
        values = input('>> ').rstrip()
        if os.path.isfile(values) == False:
            print('File not exist')
            continue
        image = load_image(name=values, size=(64, 64))
        prediction = model.predict(image)
        display(names, prediction)


if __name__ == '__main__':
    main()