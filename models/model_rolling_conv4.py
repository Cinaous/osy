import math

import numpy as np
import tensorflow.keras as kr
import tensorflow as tf
from tensorflow.keras.datasets import cifar10, mnist
from trm.gru_transformer import StandardScaler
import matplotlib.pyplot as plt


class RollingConv(kr.Model):
    def __init__(self, units, kernel_size, rolling=2, strides=2, padding='same',
                 conv=kr.layers.Conv2D, callback=None):
        super(RollingConv, self).__init__()
        unit, mod = units // rolling, units % rolling
        self.convs = [conv(unit if i != 0 else unit + mod, kernel_size, strides, padding)
                      for i in range(rolling)]
        if callable(callback):
            self.convs = [callback(conv) for conv in self.convs]

    def call(self, inputs, training=None, mask=None):
        c, n = inputs.shape[-1], len(self.convs)
        step, mod = c // n, c % n
        if step != 0:
            split_nums = [step if i != 0 else step + mod for i in range(n)]
            x = tf.split(inputs, split_nums, -1)
        else:
            x = [inputs for _ in range(n)]
        x = [conv(x_) for conv, x_ in zip(self.convs, x)]
        return tf.concat(x, -1)


cba = lambda conv: kr.Sequential([
    conv,
    kr.layers.LayerNormalization(epsilon=1e-5),
    kr.layers.LeakyReLU()
])
cbap = lambda conv: kr.Sequential([
    cba(conv),
    kr.layers.MaxPooling2D(strides=1, padding='same')
])


def unet_layer(units, kernel_size, rolling=2, strides=2, padding='same', callback=cba):
    return [units, kernel_size, rolling, strides, padding, callback]


class UNet(kr.Model):
    def __init__(self, layers, channel=3):
        """
        :param layers: units, kernel_size, rolling=2, strides=2, callback=None
        :param channel: 图片通道数
        :param callback: models.model_skip.default_layer 包装函数
        """
        super(UNet, self).__init__()
        self.downs, self.ups, n = [], [], len(layers)
        assert n > 1, 'Error: less 2 layers!'
        for i in range(n):
            units, kernel_size, rolling, strides, padding, callback = layers[i]
            down = RollingConv(units, kernel_size, rolling, strides, padding, callback=callback)
            self.downs.append(down)
            units, kernel_size, rolling, strides, padding, callback = layers[-1 - i]
            if i == n - 1:
                break
            units, _, rolling, _, _, _ = layers[-2 - i]
            up = RollingConv(units, kernel_size, rolling, strides, padding,
                             conv=kr.layers.Conv2DTranspose, callback=callback)
            self.ups.append(up)
        self.last = RollingConv(channel, kernel_size, rolling, strides, padding,
                                conv=kr.layers.Conv2DTranspose)

    def call(self, inputs, training=None, mask=None):
        x, xs = inputs, []
        for down in self.downs:
            x = down(x)
            xs.append(x)
        for up, dx in zip(self.ups, xs[::-1][1:]):
            x = up(x)
            x = tf.reduce_max([x, dx], 0)
        x = self.last(x)
        return x


if __name__ == '__main__':
    scaler = StandardScaler()
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    x_train, x_test = scaler.fit_transform(x_train), scaler.transform(x_test)
    # (x_train, y_train), (x_test, y_test) = mnist.load_data()
    # x_train, x_test = scaler.fit_transform(x_train[..., np.newaxis]), scaler.transform(x_test[..., np.newaxis])
    model = UNet([unet_layer(12, 6, 3, callback=cba),
                  unet_layer(48, 5, callback=cba)])
    xp = model(x_train[-1:])
    model.summary()
    plt.subplot(121)
    plt.imshow(xp[0])
    plt.subplot(122)
    plt.imshow(x_train[-1])
    plt.show()

    model = kr.Sequential([
        RollingConv(48, 7, 3, callback=cba),
        RollingConv(48, 7, 2, callback=cba),
        RollingConv(96, 5, 2, callback=cbap),
        RollingConv(96, 5, 2, callback=cbap),
        RollingConv(192, 3, 2, callback=cba),
        kr.layers.Flatten(),
        kr.layers.Dense(10)
    ])
    model.compile(optimizer=kr.optimizers.Adam(),
                  loss=kr.losses.SparseCategoricalCrossentropy(True),
                  metrics=[kr.metrics.sparse_categorical_accuracy])
    model.fit(x_train, y_train, 64, 5, validation_data=(x_test, y_test))
    model.summary()
