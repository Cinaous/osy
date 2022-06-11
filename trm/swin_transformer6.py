from abc import ABC
import numpy as np
import tensorflow.keras as kr
import tensorflow as tf
from tensorflow.keras.datasets import mnist, cifar10


class SwinTransformer(kr.Model, ABC):
    def __init__(self, in_channel=3, headers=8, patch_size=4, revert=False):
        super(SwinTransformer, self).__init__()
        self.stms = [SwinTransformerBlock(patch_size=patch_size, in_channel=in_channel, strides=patch_size)
                     for _ in range(headers)]
        self.last = kr.Sequential()
        self.last.add(kr.layers.Conv2DTranspose(in_channel, patch_size, patch_size, 'same')) if revert else \
            self.last.add(kr.layers.Dense(in_channel))
        self.last.add(kr.layers.LayerNormalization())
        self.last.add(kr.layers.LeakyReLU())

    def call(self, inputs, training=None, mask=None):
        outs = [stm(inputs) for stm in self.stms]
        out = tf.concat(outs, -1)
        out = self.last(out)
        return out


class SwinTransformerBlock(kr.Model, ABC):
    def __init__(self, patch_size=4, in_channel=3, strides=4):
        super(SwinTransformerBlock, self).__init__()
        self.in_channel = in_channel
        self.patch_size = patch_size
        self.convs = [(kr.layers.Conv2D(patch_size * in_channel, patch_size, strides, 'same'),
                       kr.layers.Dense(in_channel)) for _ in range(3)]

    def call(self, inputs, training=None, mask=None):
        qkv = []
        for conv, stm in self.convs:
            x = conv(inputs)
            sp = x.shape
            x = tf.reshape(x, [-1, *sp[1:-1], self.patch_size, self.in_channel])
            x = stm(x)
            qkv.append(x)
        q, k, v = qkv
        kt = tf.transpose(k, [0, 1, 2, 4, 3])
        s = tf.matmul(q, kt)
        s = tf.nn.softmax(s, axis=-1)
        x = tf.matmul(s, v)
        sp = x.shape
        x = tf.reshape(x, [-1, *sp[1:-2], sp[-2] * sp[-1]])
        return x


if __name__ == '__main__':
    # (x_train, y_train), (x_test, y_test) = mnist.load_data()
    # x_train, x_test = x_train[..., np.newaxis] / 127.5 - 1, x_test[..., np.newaxis] / 127.5 - 1
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    x_train, x_test = x_train / 127.5 - 1, x_test / 127.5 - 1
    model = kr.Sequential([
        SwinTransformer(8, 4, 4),
        kr.layers.Flatten(),
        kr.layers.Dense(10)
    ])
    model.compile(optimizer=kr.optimizers.Adam(),
                  loss=kr.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=[kr.metrics.sparse_categorical_accuracy])
    model.fit(x_train, y_train, 64, 5, validation_data=(x_test, y_test), validation_freq=1)
    model.summary()
