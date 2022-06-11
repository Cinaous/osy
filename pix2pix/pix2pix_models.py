import abc
import tensorflow.keras as kr
import tensorflow as tf
import numpy as np


class RestNet(kr.Model, abc.ABC):
    def __init__(self, rest, units, kernel_size, activation=kr.activations.swish, dropout=.2):
        super(RestNet, self).__init__()
        self.conv = [RestUnit(rest[1], units * (i + 1), kernel_size, activation, dropout) for i in range(rest[0])]

    def call(self, inputs, training=None, mask=None):
        x = inputs
        for i, conv in enumerate(self.conv):
            x = conv



class RestUnit(kr.layers.Layer):
    def __init__(self, rest, units, kernel_size, activation=kr.activations.swish, dropout=.2):
        super(RestUnit, self).__init__()
        self.conv = [(kr.layers.Conv2D(units, kernel_size, padding='same'),
                      kr.layers.LayerNormalization()) for i in range(rest)]
        self.activation = activation
        self.dropout = kr.layers.Dropout(dropout)

    def call(self, inputs, *args, **kwargs):
        x = inputs
        for i, (conv, norm) in enumerate(self.conv):
            x = conv(x)
            if i % 2 == 1:
                x += rx
            else:
                rx = x
            x = self.dropout(self.activation(norm(x)))
        return x


if __name__ == '__main__':
    x_train = np.random.uniform(size=(17, 64, 64, 3))
    net = RestUnit(7, 128, 3)
    print(net(x_train))
