from abc import ABC
import tensorflow.keras as kr
import tensorflow as tf
import numpy as np


class ConvRNN(kr.Model, ABC):
    def __init__(self, units, kernel_size, strides=2, padding='same',
                 rnn=0, conv=kr.layers.Conv2D,
                 activation=kr.activations.tanh):
        super(ConvRNN, self).__init__()
        self.conv = conv(units, kernel_size, strides, padding)
        self.norm = kr.layers.LayerNormalization(epsilon=1e-5)
        self.activation = activation
        if rnn == 0:
            self.rnn = None
        else:
            self.rnn = kr.Sequential([kr.layers.GRU(units, return_sequences=i != rnn - 1)
                                      for i in range(rnn)])

    def call(self, inputs, training=None, mask=None):
        x = self.conv(inputs)
        if self.rnn is None:
            return x if self.activation is None else self.activation(self.norm(x))
        _, h, w, c = x.shape
        r = tf.reshape(x, [-1, h * w, c])
        r = self.rnn(r)
        r = tf.expand_dims(r, 1)
        r = tf.expand_dims(r, 1)
        x += r
        return x if self.activation is None else self.activation(self.norm(x))


if __name__ == '__main__':
    x = np.random.uniform(size=[3, 128, 128, 3])
    model = ConvRNN(48, 5)
    y = model(x)
    print(y)
    model.summary()
