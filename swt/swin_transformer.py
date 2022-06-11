from abc import ABC
import tensorflow.keras as kr
import tensorflow as tf
import numpy as np


class SwinTransformer(kr.Model, ABC):
    def __init__(self):
        super(SwinTransformer, self).__init__()
        self.embedding = Embedding(48, multi_head=1)
        self.w_attention = WindowAttention(96)

    def call(self, inputs, training=None, mask=None):
        x = self.embedding(inputs)
        x = self.w_attention(x)
        return x


class Embedding(kr.Model, ABC):
    def __init__(self, dim, embed_size=4, multi_head=1):
        super(Embedding, self).__init__()
        self.conv = kr.layers.Conv2D(dim, embed_size, embed_size)
        self.dense = kr.layers.Dense(dim * multi_head)

    def call(self, inputs, training=None, mask=None):
        x = self.conv(inputs)
        x = kr.activations.gelu(x)
        x = self.dense(x)
        return x


class WindowAttention(kr.Model, ABC):
    def __init__(self, dim, window_size=7):
        super(WindowAttention, self).__init__()
        self.window_size = window_size
        self.qkv = kr.layers.Dense(3 * dim, use_bias=False)
        self.dim = dim

    def call(self, inputs, training=None, mask=None):
        _, h, w, c = inputs.shape
        assert h % self.window_size == 0 and h == w
        num = h // self.window_size
        x = tf.reshape(inputs, [-1, num, self.window_size, num, self.window_size, c])
        x = tf.transpose(x, [0, 1, 3, 2, 4, 5])
        x = tf.reshape(x, [-1, self.window_size ** 2, c])
        qkv = self.qkv(x)
        q, k, v = tf.split(qkv, 3, -1)
        qk = q @ tf.transpose(k, [0, 2, 1])
        qk = tf.nn.softmax(qk / tf.sqrt(self.dim * 1.))
        x = qk @ v
        x = tf.reshape(x, [-1, num, num, self.window_size, self.window_size, self.dim])
        x = tf.transpose(x, [0, 1, 3, 2, 4, 5])
        x = tf.reshape(x, [-1, h, w, self.dim])
        return x


if __name__ == '__main__':
    x_train = np.random.uniform(size=[23, 224, 224, 3])
    model = SwinTransformer()
    xp = model(x_train)
    print(xp.shape)
