import math
from abc import ABC
import numpy as np
import tensorflow.keras as kr
import tensorflow as tf
from tensorflow.keras.datasets import mnist, cifar10
import matplotlib.pyplot as plt


class SwinTransformerBlock(kr.Model, ABC):
    def __init__(self, units, win_size: object = 4, strides=4):
        super(SwinTransformerBlock, self).__init__()
        self.v = kr.layers.GRU(units, return_sequences=True)
        h, w = (win_size, win_size) if type(win_size) == int else win_size
        self.k = kr.layers.GRU(h * w, activation=kr.activations.softmax)
        self.strides, self.h, self.w = strides, h, w

    def call(self, inputs, training=None, mask=None):
        _, h, w, c = inputs.shape
        s, ih, iw = self.strides, self.h, self.w
        nh, hs, nw = math.ceil(h / s), [], math.ceil(w / s)
        for i in range(nh):
            ws = []
            for j in range(nw):
                mask = tf.roll(inputs, (i * s, j * s), (0, 1))
                mask = mask[:, :ih, :iw, :]
                mask = tf.reshape(mask, [-1, ih * iw, c])
                v = self.v(mask)
                k = self.k(mask)
                k = tf.expand_dims(k, 1)
                o = k @ v
                ws.append(o)
            wt = tf.concat(ws, 1)
            wt = tf.expand_dims(wt, 1)
            hs.append(wt)
        out = tf.concat(hs, 1)
        return out


if __name__ == '__main__':
    # (x_train, y_train), (x_test, y_test) = mnist.load_data()
    # x_train, x_test = x_train[..., np.newaxis] / 127.5 - 1, x_test[..., np.newaxis] / 127.5 - 1
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    x_train, x_test = x_train / 127.5 - 1, x_test / 127.5 - 1
    model = kr.Sequential([
        SwinTransformerBlock(12, 8, 8),
        kr.layers.Flatten(),
        kr.layers.Dense(10)
    ])
    # x = model(x_train[:1])
    # plt.subplot(121)
    # plt.imshow(x[0])
    # plt.subplot(122)
    # plt.imshow(x_train[0])
    # plt.show()
    model.compile(optimizer=kr.optimizers.Adam(),
                  loss=kr.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=[kr.metrics.sparse_categorical_accuracy])
    model.fit(x_train, y_train, 64, 5, validation_data=(x_test, y_test), validation_freq=1)
    model.summary()
    model.layers[0].summary()
