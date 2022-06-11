from abc import ABC
import numpy as np
import tensorflow.keras as kr
import tensorflow as tf
from tensorflow.keras.datasets import mnist, cifar10


class SwinTransformer(kr.Model, ABC):
    def __init__(self, in_channel=3, headers=8, patch_size=4, revert=False, last=True):
        super(SwinTransformer, self).__init__()
        self.stms = [SwinTransformerBlock(patch_size=patch_size,
                                          in_channel=in_channel,
                                          strides=patch_size,
                                          revert=revert)
                     for _ in range(headers)]
        if not last or revert:
            return
        self.last = kr.Sequential([
            kr.layers.Dense(in_channel),
            kr.layers.LayerNormalization(),
            kr.layers.LeakyReLU()
        ])

    def call(self, inputs, training=None, mask=None):
        outs = [stm(inputs) for stm in self.stms]
        out = tf.concat(outs, -1)
        if hasattr(self, 'last'):
            out = self.last(out)
        return out


class SwinTransformerBlock(kr.Model, ABC):
    def __init__(self, patch_size=4, in_channel=3, strides=4, revert=False):
        super(SwinTransformerBlock, self).__init__()
        self.revert = revert
        conv = kr.layers.Conv2DTranspose if revert else kr.layers.Conv2D
        self.k = conv(in_channel, patch_size, strides, 'same')
        self.v = conv(in_channel, patch_size, strides, 'same')

    def call(self, inputs, training=None, mask=None):
        k, v = self.k(inputs), self.v(inputs)
        x = tf.concat([k, v], -1) if self.revert else tf.multiply(k, v)
        return x


if __name__ == '__main__':
    # (x_train, y_train), (x_test, y_test) = mnist.load_data()
    # x_train, x_test = x_train[..., np.newaxis] / 127.5 - 1, x_test[..., np.newaxis] / 127.5 - 1
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    x_train, x_test = x_train / 127.5 - 1, x_test / 127.5 - 1
    model = kr.Sequential([
        SwinTransformer(256, 8, 4),
        kr.layers.Flatten(),
        kr.layers.Dense(10)
    ])
    model.compile(optimizer=kr.optimizers.Adam(),
                  loss=kr.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=[kr.metrics.sparse_categorical_accuracy])
    model.fit(x_train, y_train, 64, 5, validation_data=(x_test, y_test), validation_freq=1)
    model.summary()
