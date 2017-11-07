from keras.optimizers import Adam
from utils.image import load_data
from utils.file import save_model, save_list
from networks.relu import build_model


def main():
    (sets, labels, names) = load_data('images/', (64, 64))
    save_list(names, 'names.json')
    model = build_model((64, 64), 3)
    optimizer = Adam(lr=0.0001)
    save_model(model, 'model.json')
    model.compile(loss='categorical_crossentropy', optimizer=optimizer)
    model.fit(sets, labels, batch_size=3, epochs=200)
    model.save_weights('weights.hdf5')


if __name__ == '__main__':
    main()