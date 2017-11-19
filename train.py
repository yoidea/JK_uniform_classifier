from argparse import ArgumentParser
from keras.optimizers import Adam
from utils.image import load_data, to_dirname
from utils.file import save_model, save_list
from networks.relu import build_model


def get_args():
    parser = ArgumentParser()
    parser.add_argument('-i', '--input', type=str, default='images')
    parser.add_argument('-b', '--batch', type=int, default=10)
    parser.add_argument('-e', '--epoch', type=int, default=3000)
    return parser.parse_args()


def main():
    args = get_args()
    dirname = to_dirname(args.input)
    batch = args.batch
    epochs = args.epoch
    (sets, labels, names) = load_data(dirname, (64, 64))
    save_list(names, 'names.json')
    model = build_model((64, 64), 3)
    optimizer = Adam(lr=0.0001)
    save_model(model, 'model.json')
    model.compile(loss='categorical_crossentropy', optimizer=optimizer)
    model.fit(sets, labels, batch_size=batch, epochs=epochs)
    model.save_weights('weights.hdf5')


if __name__ == '__main__':
    main()