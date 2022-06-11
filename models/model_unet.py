from abc import ABC
import tensorflow.keras as kr
import tensorflow as tf
from horse2zebra.horse2zebra_datasets12 import StandardScaler
from tensorflow.keras.datasets import cifar10
import numpy as np
import matplotlib.pyplot as plt


def conv_init(channel=3, img_size=128, callback=None, regularizer=None):
    kernel_size = 1
    downs = []
    channels = [channel]
    kernel_sizes = []
    source_size = img_size
    while True:
        img_size += 1 - kernel_size
        if img_size < 2:
            break
        img_size = img_size // 2
        scale = (source_size // img_size) ** 2
        units = scale * channel
        channels.append(units)
        conv = kr.layers.Conv2D(units, kernel_size, kernel_regularizer=regularizer)
        kernel_sizes.append(kernel_size)
        pool = kr.layers.MaxPool2D()
        downs.append(callback(conv, pool, len(kernel_sizes)) if callable(callback) else kr.Sequential([conv, pool]))
        if (img_size % 2 == 1 and kernel_size % 2 == 1) or \
                (img_size % 2 == 0 and kernel_size % 2 == 0):
            kernel_size += 1
        else:
            kernel_size += 2
    return downs, channels, kernel_sizes


class UNet(kr.Model, ABC):
    def __init__(self, channel=3, img_size=128, down_callback=None, regularizer=None, up_callback=None):
        super(UNet, self).__init__()
        self.downs, channels, kernel_sizes = conv_init(channel=channel, img_size=img_size,
                                                       callback=down_callback, regularizer=regularizer)
        channels, kernel_sizes = channels[:-1][::-1], kernel_sizes[::-1]
        self.ups = []
        ix = 0
        for units, kernel_size in zip(channels, kernel_sizes):
            up = kr.layers.UpSampling2D()
            conv = kr.layers.Conv2DTranspose(units, kernel_size, kernel_regularizer=regularizer)
            ix += 1
            ix = ix if ix < len(kernel_sizes) else -1
            self.ups.append(kr.Sequential([up, up_callback(conv, ix) if callable(up_callback) and ix != -1 else conv]))

    def call(self, inputs, training=None, mask=None):
        x = inputs
        dxs = []
        for down in self.downs:
            x = down(x)
            dxs.append(x)
        dxs = dxs[:-1][::-1]
        for up, dx in zip(self.ups, dxs):
            x = up(x)
            x = tf.concat([x, dx], -1)
        x = self.ups[-1](x)
        return x


class Disc(kr.Model, ABC):
    def __init__(self, last, channel=3, img_size=128, callback=None, regularizer=None):
        super(Disc, self).__init__()
        self.downs = conv_init(channel=channel, img_size=img_size,
                               callback=callback, regularizer=regularizer)[0]
        self.last = last

    def call(self, inputs, training=None, mask=None):
        x = inputs
        for down in self.downs:
            x = down(x)
        x = self.last(x)
        return x


if __name__ == '__main__':
    scaler = StandardScaler()
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    x_train, x_test = scaler.fit_transform(x_train), scaler.fit_transform(x_test)

    model = UNet(img_size=32,
                 down_callback=lambda conv, pool, ix: kr.Sequential([
                     conv,
                     kr.layers.LeakyReLU(),
                     pool if ix < 2 else kr.Sequential([
                         pool,
                         kr.layers.Dropout(.1)
                     ])]),
                 up_callback=lambda conv, ix: kr.Sequential([
                     conv,
                     kr.layers.LeakyReLU(),
                     kr.layers.Dropout(.1)
                 ]))
    xp = model(x_train[-1:])
    model.ups[0].summary()
    plt.subplot(121)
    plt.imshow(xp[0])
    plt.subplot(122)
    plt.imshow(x_train[-1])
    plt.show()

    model = Disc(img_size=32, callback=lambda conv, pool, ix: kr.Sequential([
        conv,
        kr.layers.LeakyReLU(),
        pool if ix < 2 else kr.Sequential([
            pool,
            kr.layers.Dropout(.1)
        ])
    ]), last=kr.Sequential([
        kr.layers.Flatten(),
        kr.layers.Dense(10)
    ]))
    model.compile(optimizer=kr.optimizers.Adam(),
                  loss=kr.losses.SparseCategoricalCrossentropy(True),
                  metrics=[kr.metrics.sparse_categorical_accuracy])
    model.fit(x_train, y_train, 32, 5, validation_data=(x_test, y_test))
    model.summary()
    model.layers[0].summary()
