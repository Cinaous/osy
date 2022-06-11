from abc import ABC
import tensorflow.keras as kr
import numpy as np


def conv(container, units, kernel_size, dropout, strides, layer=kr.layers.Conv2D):
    container.add(layer(units, kernel_size, strides, 'same'))
    container.add(kr.layers.LayerNormalization())
    container.add(kr.layers.PReLU())
    container.add(kr.layers.Dropout(dropout))


class Encoder(kr.Model, ABC):
    def __init__(self, units, kernel_size, dropout=.2, n=3):
        super(Encoder, self).__init__()
        self.conv = kr.Sequential()
        for _ in range(n):
            conv(self.conv, units, kernel_size, dropout, 2)
            conv(self.conv, units * 2, kernel_size, dropout, 1)
            conv(self.conv, units, kernel_size, dropout, 1)
            kernel_size -= 2
        self.conv.add(kr.layers.Conv2D(units, kernel_size, 2, 'same'))

    def call(self, inputs, training=None, mask=None):
        return self.conv(inputs)


class Decoder(kr.Model, ABC):
    def __init__(self, units, kernel_size, dropout=.2, n=3):
        super(Decoder, self).__init__()
        self.conv = kr.Sequential()
        kernel_size -= 2 * n
        for i in range(n):
            conv(self.conv, units * 2, kernel_size, dropout, 1, kr.layers.Conv2DTranspose)
            conv(self.conv, units, kernel_size, dropout, 2, kr.layers.Conv2DTranspose)
            kernel_size += 2
        self.conv.add(kr.layers.Conv2DTranspose(3, kernel_size, 2, 'same'))

    def call(self, inputs, training=None, mask=None):
        return self.conv(inputs)


if __name__ == '__main__':
    x = np.random.uniform(size=[7, 128, 128, 3])
    model = Encoder(96, 11)
    y = model(x)
    print(y)
    model = Decoder(96, 11)
    y = model(y)
    print(y)
