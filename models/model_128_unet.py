from abc import ABC
import tensorflow.keras as kr
import tensorflow as tf
from horse2zebra.horse2zebra_datasets12 import Datasets
import numpy as np
import matplotlib.pyplot as plt


class UNet(kr.Model, ABC):
    def __init__(self):
        super(UNet, self).__init__()
        # 62 * 62 * 96
        self.conv1 = kr.Sequential([
            kr.layers.Conv2D(96, 4, 2),
            kr.layers.LayerNormalization(epsilon=1e-5),
            kr.layers.LeakyReLU(),
            kr.layers.MaxPool2D(strides=1),
            kr.layers.Dropout(.2)
        ])
        # 29 * 29 * 128
        self.conv2 = kr.Sequential([
            kr.layers.Conv2D(128, 4, 2),
            kr.layers.LayerNormalization(epsilon=1e-5),
            kr.layers.LeakyReLU(),
            kr.layers.MaxPool2D(strides=1),
            kr.layers.Dropout(.1)
        ])
        # 26 * 26 * 192
        self.conv3 = kr.Sequential([
            kr.layers.Conv2D(192, 4),
            kr.layers.LayerNormalization(epsilon=1e-5),
            kr.layers.LeakyReLU(),
            kr.layers.Dropout(.1)
        ])
        # 23 * 23 * 192
        self.conv4 = kr.Sequential([
            kr.layers.Conv2D(192, 3),
            kr.layers.LayerNormalization(epsilon=1e-5),
            kr.layers.LeakyReLU(),
            kr.layers.MaxPool2D(strides=1)
        ])
        # 20 * 20 * 192
        self.conv5 = kr.Sequential([
            kr.layers.Conv2D(192, 3),
            kr.layers.LayerNormalization(epsilon=1e-5),
            kr.layers.LeakyReLU(),
            kr.layers.MaxPool2D(strides=1)
        ])
        # 18 * 18 * 192
        self.conv6 = kr.Sequential([
            kr.layers.Conv2D(192, 3),
            kr.layers.LayerNormalization(epsilon=1e-5),
            kr.layers.LeakyReLU()
        ])
        # 16 * 16 * 192
        self.conv7 = kr.Sequential([
            kr.layers.Conv2D(192, 3),
            kr.layers.LayerNormalization(epsilon=1e-5),
            kr.layers.LeakyReLU()
        ])
        # 18 * 18 * 192
        self.conv_6 = kr.Sequential([
            kr.layers.Conv2DTranspose(192, 3),
            kr.layers.LayerNormalization(epsilon=1e-5),
            kr.layers.LeakyReLU()
        ])
        # 20 * 20 * 192
        self.conv_5 = kr.Sequential([
            kr.layers.Conv2DTranspose(192, 3),
            kr.layers.LayerNormalization(epsilon=1e-5),
            kr.layers.LeakyReLU()
        ])
        # 23 * 23 * 192
        self.conv_4 = kr.Sequential([
            kr.layers.Conv2DTranspose(192, 5),
            kr.layers.LayerNormalization(epsilon=1e-5),
            kr.layers.LeakyReLU(),
            kr.layers.MaxPool2D(strides=1)
        ])
        # 26 * 26 * 192
        self.conv_3 = kr.Sequential([
            kr.layers.Conv2DTranspose(192, 4),
            kr.layers.LayerNormalization(epsilon=1e-5),
            kr.layers.LeakyReLU(),
            kr.layers.Dropout(.1)
        ])
        # 29 * 29 * 128
        self.conv_2 = kr.Sequential([
            kr.layers.Conv2DTranspose(128, 5),
            kr.layers.LayerNormalization(epsilon=1e-5),
            kr.layers.LeakyReLU(),
            kr.layers.MaxPool2D(strides=1),
            kr.layers.Dropout(.1)
        ])
        # 62 * 62 * 96 1+ (w - k) / s = a -> (a - 1) * s +k = w -> 28 * 2 + k = 63
        self.conv_1 = kr.Sequential([
            kr.layers.Conv2DTranspose(96, 7, 2),
            kr.layers.LayerNormalization(epsilon=1e-5),
            kr.layers.LeakyReLU(),
            kr.layers.MaxPool2D(strides=1),
            kr.layers.Dropout(.2)
        ])
        # 128 * 128 * 3 61 * 2 + k = 128 - 122
        self.conv_0 = kr.layers.Conv2DTranspose(3, 6, 2)

    def call(self, inputs, training=None, mask=None):
        x1 = self.conv1(inputs)
        x2 = self.conv2(x1)
        x3 = self.conv3(x2)
        x4 = self.conv4(x3)
        x5 = self.conv5(x4)
        x6 = self.conv6(x5)
        x7 = self.conv7(x6)
        x_6 = self.conv_6(x7)
        x_5 = self.conv_5(x_6 + x6)
        x_4 = self.conv_4(x_5 + x5)
        x_3 = self.conv_3(x_4 + x4)
        x_2 = self.conv_2(x_3 + x3)
        x_1 = self.conv_1(x_2 + x2)
        x = self.conv_0(x_1 + x1)
        return x


class Dis(kr.Model, ABC):
    def __init__(self):
        super(Dis, self).__init__()
        # 62 * 62 * 96
        self.conv1 = kr.Sequential([
            kr.layers.Conv2D(96, 4, 2),
            kr.layers.LayerNormalization(epsilon=1e-5),
            kr.layers.LeakyReLU(),
            kr.layers.MaxPool2D(strides=1),
            kr.layers.Dropout(.2)
        ])
        # 29 * 29 * 128
        self.conv2 = kr.Sequential([
            kr.layers.Conv2D(128, 4, 2),
            kr.layers.LayerNormalization(epsilon=1e-5),
            kr.layers.LeakyReLU(),
            kr.layers.MaxPool2D(strides=1),
            kr.layers.Dropout(.1)
        ])
        # 26 * 26 * 192
        self.conv3 = kr.Sequential([
            kr.layers.Conv2D(192, 4),
            kr.layers.LayerNormalization(epsilon=1e-5),
            kr.layers.LeakyReLU(),
            kr.layers.Dropout(.1)
        ])
        # 23 * 23 * 192
        self.conv4 = kr.Sequential([
            kr.layers.Conv2D(192, 3),
            kr.layers.LayerNormalization(epsilon=1e-5),
            kr.layers.LeakyReLU(),
            kr.layers.MaxPool2D(strides=1)
        ])
        # 20 * 20 * 192
        self.conv5 = kr.Sequential([
            kr.layers.Conv2D(192, 3),
            kr.layers.LayerNormalization(epsilon=1e-5),
            kr.layers.LeakyReLU(),
            kr.layers.MaxPool2D(strides=1)
        ])
        # 18 * 18 * 192
        self.conv6 = kr.Sequential([
            kr.layers.Conv2D(192, 3),
            kr.layers.LayerNormalization(epsilon=1e-5),
            kr.layers.LeakyReLU()
        ])
        # 16 * 16 * 192
        self.conv7 = kr.Sequential([
            kr.layers.Conv2D(192, 3),
            kr.layers.LayerNormalization(epsilon=1e-5),
            kr.layers.LeakyReLU()
        ])
        self.conv = kr.layers.Conv2D(2, 1)

    def call(self, inputs, training=None, mask=None):
        x = self.conv1(inputs)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)
        x = self.conv7(x)
        x = self.conv(x)
        return x


if __name__ == '__main__':
    model = UNet()
    dataset = Datasets(1)
    test_horse, test_zebra = next(dataset)
    xp = model(test_horse)
    model.summary()
    plt.subplot(121)
    plt.imshow(xp[0])
    plt.subplot(122)
    plt.imshow(test_horse[0])
    plt.show()
