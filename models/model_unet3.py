from abc import ABC
import tensorflow.keras as kr
import tensorflow as tf
from horse2zebra.horse2zebra_datasets12 import StandardScaler
from tensorflow.keras.datasets import cifar10
import numpy as np
import matplotlib.pyplot as plt


def downs_initializer(size, channel, callback=None, scale=1, regularizer=None):
    w, c = size, channel
    downs, channels, ks, sizes = [], [channel], [], []
    ix = 0
    units = int(c * scale)
    while True:
        if w == size:
            kernel_size = 4
            strides = 4
            w //= 4
            units *= 16
        elif ix % 4 == 0:
            kernel_size = 2
            strides = 2
            w //= 2
            units *= 2
        else:
            kernel_size = 3 if w % 2 == 0 else 4
            strides = 1
            w -= kernel_size - 1
        if w < 1:
            break
        ix += 1
        conv = kr.layers.Conv2D(units, kernel_size, strides, kernel_regularizer=regularizer)
        channels.append(units)
        ks.append((kernel_size, strides))
        sizes.append(w)
        conv = callback(conv) if callable(callback) else conv
        downs.append(conv)
    return downs, channels, ks, sizes


class UNet(kr.Model):
    def __init__(self, size=128, channel=3, callback=None, scale=1, regularizer=None):
        super(UNet, self).__init__()
        self.downs, channels, ks, _ = downs_initializer(size, channel, callback, scale, regularizer)
        self.ups, n, i = [], len(ks), 0
        for units, (kernel_size, strides) in zip(channels[:-1][::-1], ks[::-1]):
            i += 1
            conv = kr.layers.Conv2DTranspose(units, kernel_size, strides, kernel_regularizer=regularizer)
            conv = callback(conv) if callable(callback) and i != n else conv
            self.ups.append(conv)
        self.build((None, size, size, channel))

    def call(self, inputs, training=None, mask=None):
        x, dxs = inputs, []
        for down in self.downs:
            x = down(x)
            dxs.append(x)
        for up, dx in zip(self.ups, dxs[:-1][::-1]):
            x = up(x)
            x += dx
        x = self.ups[-1](x)
        return x


class Discriminator(kr.Model):
    def __init__(self, size=128, classifer=(12, 2), channel=3, callback=None, scale=1, regularizer=None):
        super(Discriminator, self).__init__()
        self.downs, channels, _, sizes = downs_initializer(size, channel, callback, scale, regularizer)
        input_shape = [None, size, size, channel]
        w, c = sizes[-1], channels[-1]
        self.hsape = self.add_weight('hsape', [w, c, classifer[0]], dtype=self.dtype,
                                    initializer='glorot_uniform', trainable=True)
        self.wsape = self.add_weight('wsape', [w, c, classifer[0]], dtype=self.dtype,
                                    initializer='glorot_uniform', trainable=True)
        self.csape = self.add_weight('csape', [c, classifer[1]], dtype=self.dtype,
                                    initializer='glorot_uniform', trainable=True)
        self.build(input_shape)

    def call(self, inputs, training=None, mask=None):
        x = inputs
        for down in self.downs:
            x = down(x)
        x = tf.einsum('...hwc,hci->...iwc', x, self.hsape)
        x = tf.nn.leaky_relu(x)
        x = tf.einsum('...iwc,wcj->...ijc', x, self.wsape)
        x = tf.nn.leaky_relu(x)
        x = tf.einsum('...hwc,co->...hwo', x, self.csape)
        return x


if __name__ == '__main__':
    scaler = StandardScaler()
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    x_train, x_test = scaler.fit_transform(x_train), scaler.fit_transform(x_test)
    model = UNet(32, callback=lambda conv: kr.Sequential([
        conv,
        kr.layers.LayerNormalization(),
        kr.layers.LeakyReLU()
    ]))
    model.summary()
    xp = model(x_train[-1:])
    plt.subplot(121)
    plt.imshow(x_train[-1])
    plt.subplot(122)
    plt.imshow(xp[0])
    plt.show()

    model = kr.Sequential([
        Discriminator(32, callback=lambda conv: kr.Sequential([
            conv,
            kr.layers.LeakyReLU()
        ])),
        kr.layers.Flatten(),
        kr.layers.Dense(10)
    ])
    model.layers[0].summary()
    model.compile(optimizer=kr.optimizers.Adam(),
                  loss=kr.losses.SparseCategoricalCrossentropy(True),
                  metrics=[kr.metrics.sparse_categorical_accuracy])
    model.fit(x_train, y_train, 32, 5, validation_data=(x_test, y_test))
    model.layers[0].summary()
