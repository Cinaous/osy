from abc import ABC
import numpy as np
import tensorflow.keras as kr
import tensorflow as tf
from tensorflow.keras.datasets import mnist


class SwinTransformer(kr.Model, ABC):
    def __init__(self, layers, patch_size=4, in_channel=3, revert=False):
        super(SwinTransformer, self).__init__()
        self.stm = SwinTransformerBlock(patch_size=patch_size, in_channel=in_channel, strides=patch_size)
        self.stms = [
            SwinTransformerBlock(patch_size=patch_size, in_channel=2 ** (layer + 1) * in_channel,
                                 strides=1) for layer in range(layers)]
        if revert:
            self.last = kr.layers.Conv2DTranspose(2 ** layers * in_channel, patch_size, patch_size, 'same')

    def call(self, inputs, training=None, mask=None):
        x = self.stm(inputs)
        for stm in self.stms:
            x = stm(x)
        if hasattr(self, 'last'):
            x = self.last(x)
        return x


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
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train, x_test = x_train[..., np.newaxis] / 127.5 - 1, x_test[..., np.newaxis] / 127.5 - 1
    model = kr.Sequential([
        # 2 ** layers * in_channel
        # SwinTransformer(1, in_channel=1, revert=True),
        SwinTransformer(2, in_channel=1),
        # 2 * 4 ** 2 * 2
        # SwinTransformer(1, in_channel=64),
        # in_channel * patch_size ** 2 * 2 ** (layers + 1)
        # 64 * 4 ** 2 * 2 ** (1 + 1)
        # SwinTransformer(0, in_channel=2048, patch_size=2),
        kr.layers.Flatten(),
        kr.layers.Dense(10)
    ])
    model.compile(optimizer=kr.optimizers.Adam(),
                  loss=kr.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=[kr.metrics.sparse_categorical_accuracy])
    model.fit(x_train, y_train, 64, 3, validation_data=(x_test, y_test), validation_freq=1)
    model.summary()
