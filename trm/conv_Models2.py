import tensorflow.keras as kr
import os.path as path
from tensorflow.keras.datasets import cifar10


def sample(features, kernel_size, strides, drop=None):
    model = kr.Sequential([
        kr.layers.Conv2D(features, kernel_size, strides, padding='same'),
        kr.layers.BatchNormalization(),
        kr.layers.LeakyReLU(),
        kr.layers.MaxPool2D(padding='same')
    ])
    if drop is not None:
        model.add(kr.layers.Dropout(drop))
    return model


def samples(params, last_features):
    model = kr.Sequential()
    for features, kernel_size, strides, drop in params:
        model.add(sample(features, kernel_size, strides, drop))

    model.add(kr.layers.Flatten())
    for feature in last_features[:-1]:
        model.add(kr.Sequential([
            kr.layers.Dense(feature, kernel_regularizer=kr.regularizers.l2()),
            kr.layers.BatchNormalization(),
            kr.layers.LeakyReLU()
        ]))
    model.add(kr.layers.Dense(last_features[-1]))
    return model


if __name__ == '__main__':
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    x_train, x_test = x_train / 127.5 - 1., x_test / 127.5 - 1.
    model = samples([(64, 5, 2, None),
                     (128, 3, 2, .2),
                     (512, 3, 2, .5)], [256, 128, 10])
    model.compile(optimizer=kr.optimizers.Adam(),
                  loss=kr.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=[kr.metrics.sparse_categorical_accuracy])
    model_save_path = 'samples2.ckpt'
    model_cp = kr.callbacks.ModelCheckpoint(model_save_path, save_weights_only=True, save_best_only=True,
                                            monitor='sparse_categorical_accuracy')
    if path.exists(model_save_path + '.index'):
        model.load_weights(model_save_path)
    model.fit(x_train, y_train, 64, 2000, validation_data=(x_test, y_test), validation_freq=1,
              callbacks=[model_cp])
    model.summary()
