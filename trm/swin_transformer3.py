from abc import ABC
import numpy as np
import tensorflow.keras as kr
import tensorflow as tf
from tensorflow.keras.datasets import mnist, cifar10


class SwinTransformer(kr.Model, ABC):
    def __init__(self, patch_size=4, in_channel=3, revert=False):
        super(SwinTransformer, self).__init__()
        self.stm = SwinTransformerBlock(patch_size=patch_size, in_channel=in_channel, strides=patch_size)
        self.stmr = SwinTransformerBlock(patch_size=patch_size, in_channel=in_channel, strides=1)
        if revert:
            self.last = kr.layers.Conv2DTranspose(in_channel, patch_size, patch_size, 'same')

    def call(self, inputs, training=None, mask=None):
        x = self.stm(inputs)
        x = self.stmr(x)
        if hasattr(self, 'last'):
            x = self.last(x)
        return x


class SwinTransformers(kr.Model, ABC):
    def __init__(self, in_channel, headers, patch_size=4, revert=False):
        super(SwinTransformers, self).__init__()
        self.trms = [SwinTransformer(patch_size, in_channel, revert) for _ in range(headers)]
        self.d = kr.Sequential([
            kr.layers.Dense(in_channel, kernel_regularizer=kr.regularizers.l2()),
            kr.layers.LayerNormalization(),
            kr.layers.LeakyReLU()
        ])

    def call(self, inputs, training=None, mask=None):
        outs = [trm(inputs) for trm in self.trms]
        out = tf.concat(outs, axis=-1)
        out = self.d(out)
        return out


class SwinTransformerBlock(kr.Model, ABC):
    def __init__(self, patch_size=4, in_channel=3, strides=4):
        super(SwinTransformerBlock, self).__init__()
        self.in_channel = in_channel
        self.seq = patch_size ** 2
        self.conv = kr.layers.Conv2D(self.seq * in_channel, patch_size, strides, 'same')
        self.q = kr.layers.Dense(in_channel * 2)
        self.k = kr.layers.Dense(in_channel * 2)
        self.v = kr.layers.Dense(in_channel * 2)

    def call(self, inputs, training=None, mask=None):
        x = self.conv(inputs)
        sp = x.shape
        x = tf.reshape(x, [-1, *sp[1:-1], self.seq, self.in_channel])
        q = self.q(x)
        k = self.k(x)
        v = self.v(x)
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
        # patch_size ** 2 * 2 * in_channel
        SwinTransformers(4, 4),
        kr.layers.Flatten(),
        kr.layers.Dense(10)
    ])
    model.compile(optimizer=kr.optimizers.Adam(),
                  loss=kr.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=[kr.metrics.sparse_categorical_accuracy])
    model.fit(x_train, y_train, 64, 5, validation_data=(x_test, y_test), validation_freq=1)
    model.summary()
