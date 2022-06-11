import tensorflow as tf
import tensorflow.keras as kr
import numpy as np


def attention(query, key, value, bias=None):
    score = tf.einsum('...ij,...kj->...ik', query, key)
    score = score / np.sqrt(query.shape[-1])
    score = score if bias is None else score + bias
    score = tf.nn.softmax(score)
    attn = score @ value
    return attn


class TransformerBlock(kr.layers.Layer):
    def __init__(self, window_size, num_heads=8):
        super(TransformerBlock, self).__init__()
        self.bias_embed = None
        self.qkv = None
        self.window_size = window_size
        self.num_heads = num_heads

    def call(self, inputs, *args, **kwargs):
        _, h, w, c = inputs.shape
        assert h % self.window_size == 0
        assert w % self.window_size == 0
        assert c % self.num_heads == 0
        self.qkv = self.qkv or kr.layers.Dense(3 * c, kr.activations.gelu)
        nh, nw, nd = h // self.window_size, w // self.window_size, c // self.num_heads
        self.bias_embed = self.bias_embed or kr.layers.Embedding(self.window_size ** 2, nd)
        x = tf.reshape(inputs, [-1, nh, self.window_size, nw, self.window_size, c])
        x = tf.transpose(x, [0, 1, 3, 2, 4, 5])
        x = tf.reshape(x, [-1, self.window_size ** 2, c])
        qkv = self.qkv(x)
        qkv = tf.reshape(qkv, [-1, self.window_size ** 2, 3, self.num_heads, nd])
        bias = self.bias_embed(np.arange(self.window_size ** 2))
        bias = bias @ tf.transpose(bias, [1, 0])
        qkv = tf.transpose(qkv, [0, 2, 3, 1, 4])
        q, k, v = tf.unstack(qkv, axis=1)
        x = attention(q, k, v, bias[np.newaxis, np.newaxis, ...])
        x = tf.reshape(x, [-1, nh * nw, self.num_heads, self.window_size ** 2, nd])
        shifted_x = tf.roll(x, -1, 1)
        x = attention(x, shifted_x, shifted_x, bias[np.newaxis, np.newaxis, np.newaxis, ...])
        x = tf.transpose(x, [0, 1, 3, 2, 4])
        x = tf.reshape(x, [-1, nh, nw, self.window_size, self.window_size, c])
        x = tf.transpose(x, [0, 1, 3, 2, 4, 5])
        x = tf.reshape(x, [-1, h, w, c]) + inputs
        return x


class Transformer(kr.layers.Layer):
    def __init__(self, n, units, window_size, patch_size, num_heads=8, dropout=.2):
        super(Transformer, self).__init__()
        self.block = [TransformerBlock(window_size, num_heads) for _ in range(n)]
        self.conv = [kr.Sequential([
            kr.layers.Conv2D(units * 2 ** i, patch_size, 2, 'same'),
            kr.layers.LayerNormalization(epsilon=1e-5),
            kr.layers.PReLU(),
            kr.layers.Dropout(dropout)
        ]) for i in range(n)]

    def call(self, inputs, *args, **kwargs):
        x = inputs
        for conv, block in zip(self.conv, self.block):
            x = conv(x)
            x = block(x)
        return x


class TransformerTranspose(kr.layers.Layer):
    def __init__(self, n, units, window_size, patch_size, num_heads=8, dropout=.2):
        super(TransformerTranspose, self).__init__()
        self.block = [TransformerBlock(window_size, num_heads) for _ in range(1, n)]
        self.conv = None
        self.n, self.patch_size, self.dropout = n, patch_size, dropout
        self.last = kr.Sequential([
            kr.layers.LayerNormalization(epsilon=1e-5),
            kr.layers.Conv2DTranspose(units, patch_size, 2, 'same')
        ])

    def call(self, inputs, *args, **kwargs):
        c = inputs.shape[-1]
        self.conv = self.conv or [kr.Sequential([
            kr.layers.Conv2DTranspose(c // 2 ** i, self.patch_size, 2, 'same'),
            kr.layers.LayerNormalization(epsilon=1e-5),
            kr.layers.PReLU(),
            kr.layers.Dropout(self.dropout)
        ]) for i in range(1, self.n)]
        x = inputs
        for conv, block in zip(self.conv, self.block):
            x = conv(x)
            x = block(x)
        x = self.last(x)
        return x


if __name__ == '__main__':
    x_train = np.random.uniform(size=[3, 128, 128, 3])
    # 64, 32, 16, 8
    model = Transformer(4, 96, 8, 5)
    y_train = model(x_train)
    print(y_train)
    model = kr.Sequential([
        Transformer(4, 96, 8, 5),
        TransformerTranspose(4, 3, 8, 5)
    ])
    y_train = model(x_train)
    print(y_train)
