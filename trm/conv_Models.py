import numpy as np
import tensorflow.keras as kr
import os.path as path
from tensorflow.keras.datasets import mnist


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
    model.add(kr.layers.Dense(last_features, kernel_regularizer=kr.regularizers.l2()))
    return model


if __name__ == '__main__':
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train, x_test = x_train[..., np.newaxis] / 127.5 - 1., x_test[..., np.newaxis] / 127.5 - 1.
    model = samples([(32, 5, 2, None),
                     (128, 3, 2, .2),
                     (512, 3, 2, .5)], 10)
    model.compile(optimizer=kr.optimizers.Adam(),
                  loss=kr.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=[kr.metrics.sparse_categorical_accuracy])
    model_save_path = 'samples.ckpt'
    model_cp = kr.callbacks.ModelCheckpoint(model_save_path, save_weights_only=True, save_best_only=True,
                                            monitor='sparse_categorical_accuracy')
    if path.exists(model_save_path + '.index'):
        model.load_weights(model_save_path)
    model.fit(x_train, y_train, 64, 2000, validation_data=(x_test, y_test), validation_freq=1,
              callbacks=[model_cp])
    model.summary()
