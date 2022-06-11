import random
from abc import ABC
import numpy as np
import tensorflow.keras as kr
import tensorflow as tf
from tensorflow.keras.datasets import mnist, cifar10


class SwinTransformerBlock(kr.Model, ABC):
    def __init__(self, units=3, return_sequences=True):
        super(SwinTransformerBlock, self).__init__()
        self.gru = kr.layers.GRU(units, return_sequences=return_sequences)
        self.return_sequences = return_sequences

    def call(self, inputs, training=None, mask=None):
        xs = []
        for x in tf.split(inputs, inputs.shape[-1], -1):
            x = tf.squeeze(x, -1)
            x = self.gru(x)
            xs.append(x)
        x = tf.concat(xs, -1)
        if self.return_sequences:
            x = x[..., tf.newaxis]
        return x


if __name__ == '__main__':
    # (x_train, y_train), (x_test, y_test) = mnist.load_data()
    # x_train, x_test = x_train[..., np.newaxis] / 127.5 - 1, x_test[..., np.newaxis] / 127.5 - 1
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    x_train, x_test = x_train / 127.5 - 1, x_test / 127.5 - 1
    model = kr.Sequential([
        SwinTransformerBlock(128),
        SwinTransformerBlock(64, False),
        kr.layers.Dense(10)
    ])
    model.compile(optimizer=kr.optimizers.Adam(),
                  loss=kr.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=[kr.metrics.sparse_categorical_accuracy])
    model.fit(x_train, y_train, 64, 5, validation_data=(x_test, y_test), validation_freq=1)
    model.summary()
