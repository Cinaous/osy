import tensorflow.keras as kr
import tensorflow as tf
import numpy as np


class MultiHeadSelfAttention(kr.layers.Layer):
    def __init__(self, window_size, num_heads=8):
        super(MultiHeadSelfAttention, self).__init__()
        self.num_heads = num_heads
        self.window_size = window_size
        self.qkv = None

    def call(self, inputs, *args, **kwargs):
        _, h, w, c = inputs.shape
        self.qkv = self.qkv or kr.layers.Dense(3 * c, use_bias=False)
        qkv = self.qkv(inputs)
        assert c % self.num_heads == 0
        assert h % self.window_size == 0
        assert w % self.window_size == 0
        nh, nw, nd = h // self.window_size, w // self.window_size, c // self.num_heads
        qkv = tf.reshape(qkv, [-1, nh, self.window_size,
                               nw, self.window_size,
                               3, self.num_heads, nd])
        qkv = tf.transpose(qkv, [0, 5, 6, 1, 3, 2, 4, 7])
        qkv = tf.reshape(qkv, [-1, 3, self.num_heads, nh * nw, self.window_size ** 2, nd])
        q, k, v = tf.unstack(qkv, axis=1)
        kt = tf.transpose(k, [0, 1, 2, 4, 3])
        attn = q @ kt
        attn = tf.nn.softmax(attn / tf.sqrt(float(nd)), -1)
        attn = attn @ v
        attn = tf.reshape(attn, [-1, self.num_heads, nh, nw, self.window_size, self.window_size, nd])
        attn = tf.transpose(attn, [0, 2, 4, 3, 5, 1, 6])
        attn = tf.reshape(attn, [-1, h, w, c])
        return attn


class PatchEmbed(kr.layers.Layer):
    def __init__(self, embed_dim=96, patch_size=4):
        super(PatchEmbed, self).__init__()
        self.conv = kr.layers.Conv2D(embed_dim, patch_size, patch_size)

    def call(self, inputs, *args, **kwargs):
        return self.conv(inputs)


class Transformer(kr.layers.Layer):
    def __init__(self, window_feature, num_head=8, hidden_scale=2,
                 embed_dim=96, patch_size=4, dropout=.2):
        """
        # 32, 16, 8
        model = Transformer([((8, 16), 1), ((4, 8), 3), ((2, 4), 1)])
        x_train = np.random.uniform(size=(7, 128, 128, 3))
        :param window_feature: list, 窗口和重复次数的组合
        :param num_head:
        :param hidden_scale:
        :param embed_dim:
        :param patch_size:
        """
        super(Transformer, self).__init__()
        self.embed = PatchEmbed(embed_dim, patch_size)
        self.block = []
        for window_size, n in window_feature[:-1]:
            self.block.append(kr.Sequential([
                kr.layers.Dropout(dropout),
                kr.layers.LayerNormalization(),
                *[TransformerBlock(window_size, num_head, hidden_scale) for _ in range(n)],
                kr.layers.Dropout(dropout),
                kr.layers.LayerNormalization(),
                PatchMerging(np.sum(window_size) // len(window_size))
            ]))
        window_size, n = window_feature[-1]
        self.block.append(kr.Sequential([
            kr.layers.Dropout(dropout),
            kr.layers.LayerNormalization(),
            *[TransformerBlock(window_size, num_head, hidden_scale) for _ in range(n)]
        ]))

    def call(self, inputs, *args, **kwargs):
        x = self.embed(inputs)
        for block in self.block:
            x = block(x)
        return x


class PatchMerging(kr.layers.Layer):
    def __init__(self, kernel_size):
        super(PatchMerging, self).__init__()
        self.conv, self.kernel_size = None, int(kernel_size)

    def call(self, inputs, *args, **kwargs):
        dim = inputs.shape[-1]
        self.conv = self.conv or kr.layers.Conv2D(dim, self.kernel_size, 2, 'same')
        x = self.conv(inputs)
        return x


class TransformerBlock(kr.layers.Layer):
    def __init__(self, window_sizes, num_head=8, hidden_scale=2, dropout=.2):
        super(TransformerBlock, self).__init__()
        self.attn, self.norm, self.hidden, self.hidden_scale = [], [], None, hidden_scale
        self.dropout = dropout
        for window_size in window_sizes:
            self.attn.append(MultiHeadSelfAttention(window_size, num_head))
            self.norm.append(kr.Sequential([
                kr.layers.Dropout(dropout),
                kr.layers.LayerNormalization()
            ]))

    def call(self, inputs, *args, **kwargs):
        _, h, w, c = inputs.shape
        self.hidden = self.hidden or [kr.Sequential([
            kr.layers.Dropout(self.dropout),
            kr.layers.LayerNormalization(),
            kr.layers.Dense(self.hidden_scale * c, kr.activations.gelu),
            kr.layers.Dropout(self.dropout),
            kr.layers.LayerNormalization(),
            kr.layers.Dense(c)
        ]) for _ in range(len(self.norm))]
        x = inputs
        for norm, attn, hidden in zip(self.norm, self.attn, self.hidden):
            shortcut = x = norm(x)
            x = attn(x)
            x += shortcut
            x = hidden(x)
        return x


if __name__ == '__main__':
    # 32, 16, 8
    model = Transformer([((8, 16), 1), ((4, 8), 3), ((2, 4), 1)])
    x_train = np.random.uniform(size=(7, 128, 128, 3))
    y = model(x_train)
    print(y)
