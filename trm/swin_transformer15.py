from abc import ABC
import numpy as np
import tensorflow.keras as kr
import tensorflow as tf
from tensorflow.keras.datasets import mnist, cifar10
import matplotlib.pyplot as plt


class BlockAttribute:
    def __init__(self, units, win_size=4, strides=4, padding='same',
                 norm=None, activation=None, pooling=None, drop=None):
        self.units = units
        self.win_size = win_size
        self.strides = strides
        self.padding = padding
        self.norm = norm
        self.activation = activation
        self.pooling = pooling
        self.drop = drop

    def block_args(self, revert=False):
        return self.units, self.win_size, self.strides, self.padding, \
               kr.layers.Conv2DTranspose if revert else kr.layers.Conv2D, \
               self.norm, self.activation, self.pooling, self.drop


class SwinUNet(kr.Model, ABC):
    def __init__(self, layers: [BlockAttribute], channel=3):
        super(SwinUNet, self).__init__()
        self.downs = [SwinTransformerBlock(*args.block_args()) for args in layers]
        self.ups = [SwinTransformerBlock(*args.block_args(True)) for args in layers[:-1][::-1]]
        args = layers[0]
        self.last = SwinTransformerBlock(channel, args.win_size, args.strides, args.padding,
                                         conv=kr.layers.Conv2DTranspose)

    def call(self, inputs, training=None, mask=None):
        downs, x = [], inputs
        for down in self.downs:
            x = down(x)
            downs.append(x)
        for up, down_x in zip(self.ups, downs[:-1][::-1]):
            x = up(x) + down_x
        x = self.last(x)
        return x


class SwinTransformer(kr.Model, ABC):
    def __init__(self, layers: [BlockAttribute], channel=3):
        super(SwinTransformer, self).__init__()
        self.downs = [SwinTransformerBlock(*args.block_args()) for args in layers]
        self.nu = len(layers) // 2
        layers = layers[::-1][1:]
        self.ups = [SwinTransformerBlock(*args.block_args(True)) for args in layers[:self.nu]]
        args = layers[self.nu]
        self.last = SwinTransformerBlock(channel, args.win_size, args.strides, args.padding,
                                         conv=kr.layers.Conv2DTranspose)

    def call(self, inputs, training=None, mask=None):
        downs, x = [], inputs
        for down in self.downs:
            x = down(x)
            downs.append(x)
        for up, down_x in zip(self.ups, downs[::-1][1:self.nu + 1]):
            x = up(x) + down_x
        x = self.last(x)
        return x


class SwinTransformerBlock(kr.Model, ABC):
    def __init__(self, units, win_size, strides=4, padding='same', conv=kr.layers.Conv2D,
                 norm=None, activation=None, pooling=None, drop=None):
        super(SwinTransformerBlock, self).__init__()
        h, w = (win_size, win_size) if type(win_size) == int else win_size
        self.win_area, self.units = h * w, units
        self.v = kr.Sequential([conv(self.win_area * units, win_size, strides, padding)])
        self.k = kr.Sequential([conv(self.win_area, win_size, strides, padding)])
        if norm is not None:
            self.v.add(norm)
        if activation is not None:
            self.v.add(kr.layers.Activation(activation))
        if pooling is not None:
            self.v.add(pooling)
            self.k.add(pooling)
        if drop is not None:
            self.v.add(drop)
        self.k.add(kr.layers.Softmax())

    def call(self, inputs, training=None, mask=None):
        v = self.v(inputs)
        v = tf.reshape(v, [-1, *v.shape[1:-1], self.win_area, self.units])
        k = self.k(inputs)
        k = tf.expand_dims(k, 3)
        x = k @ v
        x = tf.squeeze(x, 3)
        return x


if __name__ == '__main__':
    # (x_train, y_train), (x_test, y_test) = mnist.load_data()
    # x_train, x_test = x_train[..., np.newaxis] / 127.5 - 1, x_test[..., np.newaxis] / 127.5 - 1
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    x_train, x_test = x_train / 127.5 - 1, x_test / 127.5 - 1
    # 16 8 4
    model = kr.Sequential([
        SwinTransformer([BlockAttribute(12, strides=2),
                         BlockAttribute(48, strides=2),
                         BlockAttribute(96, strides=2)]),
        kr.layers.Flatten(),
        kr.layers.Dense(10)
    ])
    # xx = model(x_train[:1])
    # print(xx.shape)
    # plt.subplot(121)
    # plt.imshow(xx[0])
    # plt.subplot(122)
    # plt.imshow(x_train[0])
    # plt.show()
    model.compile(optimizer=kr.optimizers.Adam(),
                  loss=kr.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=[kr.metrics.sparse_categorical_accuracy])
    model.fit(x_train, y_train, 64, 5, validation_data=(x_test, y_test), validation_freq=1)
    model.summary()
    model.layers[0].summary()
