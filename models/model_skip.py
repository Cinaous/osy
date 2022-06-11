import numpy as np
import tensorflow.keras as kr
import tensorflow as tf
from trm.gru_transformer import StandardScaler
from tensorflow.keras.datasets import cifar10
import matplotlib.pyplot as plt
import cv2


def default_layer(i, conv, up=False, iter=2, epsilon=1e-5,
                  norm=kr.layers.LayerNormalization,
                  activation=tf.nn.leaky_relu,
                  pooling=kr.layers.MaxPooling2D,
                  drop=None):
    if i % iter != 0:
        return conv
    model = kr.Sequential([conv])
    if norm is not None:
        model.add(norm(epsilon=epsilon))
    if activation is not None:
        model.add(kr.layers.Activation(activation))
    if pooling is not None:
        model.add(pooling(strides=1, padding='same'))
    if drop is not None:
        model.add(kr.layers.Dropout(drop))
    return model


class UNet(kr.Model):
    def __init__(self, layers, channel=3, callback=default_layer):
        """
        :param layers: units, kernel_size, strides
        :param channel: 图片通道数
        :param callback: models.model_skip.default_layer 包装函数
        """
        super(UNet, self).__init__()
        self.downs, self.ups, n = [], [], len(layers)
        assert n > 1, 'Error: less 2 layers!'
        for i in range(n):
            units, kernel_size, strides = layers[i]
            down = kr.layers.Conv2D(units, kernel_size, strides, 'same')
            if callable(callback):
                down = callback(i, down)
            self.downs.append(down)
            units, kernel_size, strides = layers[-1 - i]
            if i == n - 1:
                break
            units, _, _ = layers[-2 - i]
            up = kr.layers.Conv2DTranspose(units, kernel_size, strides, 'same')
            if callable(callback):
                up = callback(i, up, True)
            dense = kr.layers.Dense(1)
            self.ups.append((up, dense))
        self.last = kr.layers.Conv2DTranspose(channel, kernel_size, strides, 'same')

    def call(self, inputs, training=None, mask=None):
        x, xs = inputs, []
        for down in self.downs:
            x = down(x)
            xs.append(x)
        for (up, dense), dx in zip(self.ups, xs[::-1][1:]):
            x = up(x)
            x = tf.stack([x, dx], -1)
            x = dense(x)
            x = tf.squeeze(x, -1)
        x = self.last(x)
        return x


class RnnModel(kr.Model):
    def __init__(self, units, size, activation=None):
        super(RnnModel, self).__init__()
        h, w = (size, size) if type(size) == int else size
        self.h = kr.layers.Dense(h, activation=activation)
        self.w = kr.layers.Dense(w, activation=activation)
        self.units = kr.layers.Dense(units, activation=activation)

    def call(self, inputs, training=None, mask=None):
        x = tf.transpose(inputs, [0, 2, 3, 1])
        x = self.h(x)
        x = tf.transpose(x, [0, 3, 2, 1])
        x = self.w(x)
        x = tf.transpose(x, [0, 1, 3, 2])
        x = self.units(x)
        return x


class RnnUNet(kr.Model):
    def __init__(self, layers, im_size, channel=3, callback=default_layer):
        """
        :param layers: units, size, activation
        :param channel: 图片通道数
        :param callback: models.model_skip.default_layer 包装函数
        """
        super(RnnUNet, self).__init__()
        self.downs, self.ups, n = [], [], len(layers)
        assert n > 1, 'Error: less 2 layers!'
        for i in range(n):
            units, size, activation = layers[i]
            down = RnnModel(units, size, activation)
            if callable(callback):
                down = callback(i, down)
            self.downs.append(down)
            if i == n - 1:
                break
            units, size, activation = layers[-2 - i]
            up = RnnModel(units, size, activation)
            if callable(callback):
                up = callback(i, up, True)
            dense = kr.layers.Dense(1)
            self.ups.append((up, dense))
        self.last = RnnModel(channel, im_size)

    def call(self, inputs, training=None, mask=None):
        x, xs = inputs, []
        for down in self.downs:
            x = down(x)
            xs.append(x)
        for (up, dense), dx in zip(self.ups, xs[::-1][1:]):
            x = up(x)
            x = tf.stack([x, dx], -1)
            x = dense(x)
            x = tf.squeeze(x, -1)
        x = self.last(x)
        return x


if __name__ == '__main__':
    scaler = StandardScaler()
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    x_train, x_test = scaler.fit_transform(x_train), scaler.transform(x_test)

    img = x_test[-15]
    h, w, c = img.shape
    m = cv2.getRotationMatrix2D((h // 2, w // 2), 15, 1)
    img_ = cv2.warpAffine(img, m, (h, w))
    plt.subplot(121)
    plt.imshow(img)
    plt.subplot(122)
    plt.imshow(img_)
    plt.show()

    model = UNet([(12, 5, 2), (48, 3, 2), (96, 3, 2)])
    # model = RnnModel(3, 24, tf.nn.leaky_relu)
    # activation, (trainA, w) = tf.nn.leaky_relu, x_train.shape[1:-1]
    # model = RnnUNet([(48 * 2 ** i, (trainA // 2 // 2 ** i, w // 2 // 2 ** i), activation) for i in range(4)], (trainA, w))

    xp = model(x_train[-1:])
    plt.subplot(121)
    plt.imshow(x_train[-1])
    plt.subplot(122)
    plt.imshow(xp[0])
    plt.show()
    model.summary()

    model = kr.Sequential([model, kr.layers.Flatten(),
                           kr.layers.Dense(10)])
    model.compile(optimizer=kr.optimizers.Adam(),
                  loss=kr.losses.SparseCategoricalCrossentropy(True),
                  metrics=[kr.metrics.sparse_categorical_accuracy])
    model.fit(x_train, y_train, 32, 5, validation_data=(x_test, y_test))
    model.summary()
