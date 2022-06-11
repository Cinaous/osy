from abc import ABC
import numpy as np
import tensorflow.keras as kr
import tensorflow as tf
from tensorflow.keras.datasets import mnist, cifar10


class SwinTransformerBlock(kr.Model, ABC):
    def __init__(self, size):
        super(SwinTransformerBlock, self).__init__()
        self.xy, self.size = None, size

    def call(self, inputs, training=None, mask=None):
        if self.xy is None:
            self.xy = [(kr.layers.GRU(self.size, return_sequences=True),
                        kr.layers.GRU(self.size, return_sequences=True))
                       for _ in range(inputs.shape[-1])]
        i, outs = 0, []
        for x, y in self.xy:
            out = inputs[..., i]
            out = x(out)
            out = tf.transpose(out, [0, 2, 1])
            out = y(out)
            outs.append(out[..., tf.newaxis])
            i += 1
        out = tf.concat(outs, -1)
        return out


if __name__ == '__main__':
    # (x_train, y_train), (x_test, y_test) = mnist.load_data()
    # x_train, x_test = x_train[..., np.newaxis] / 127.5 - 1, x_test[..., np.newaxis] / 127.5 - 1
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    x_train, x_test = x_train / 127.5 - 1, x_test / 127.5 - 1
    model = kr.Sequential([
        SwinTransformerBlock(32),
        kr.layers.Flatten(),
        kr.layers.Dense(10)
    ])
    model.compile(optimizer=kr.optimizers.Adam(),
                  loss=kr.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=[kr.metrics.sparse_categorical_accuracy])
    model.fit(x_train, y_train, 64, 5, validation_data=(x_test, y_test), validation_freq=1)
    model.summary()
    model.layers[0].summary()
