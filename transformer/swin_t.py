from abc import ABC
import numpy as np
import tensorflow.keras as kr
import tensorflow as tf


class PatchEmbed(kr.Model, ABC):
    def __init__(self, embed_dim=96, patch_size=4, normalize=kr.layers.LayerNormalization):
        super(PatchEmbed, self).__init__()
        self.proj = kr.layers.Conv2D(embed_dim, patch_size, patch_size)
        self.norm = None if normalize is None else normalize()

    def call(self, inputs, training=None, mask=None):
        x = self.proj(inputs)
        # _, trainA, w, c = x.shape
        # x = tf.reshape(x, [-1, trainA * w, c])
        if self.norm is not None:
            return self.norm(x)
        return x


class PatchMerging(kr.Model, ABC):
    def __init__(self, normalize=kr.layers.LayerNormalization):
        super(PatchMerging, self).__init__()
        self.norm = normalize()
        self.dense = None

    def call(self, inputs, training=None, mask=None):
        _, h, w, c = inputs.shape
        if self.dense is None:
            self.dense = kr.layers.Dense(2 * c, use_bias=False)
        x1 = inputs[:, ::2, ::2, :]
        x2 = inputs[:, ::2, 1::2, :]
        x3 = inputs[:, 1::2, ::2, :]
        x4 = inputs[:, 1::2, 1::2, :]
        x = tf.concat([x1, x2, x3, x4], -1)
        x = self.norm(x)
        x = self.dense(x)
        return x


def window_partition(x, window_size, ut=tf):
    _, h, w, c = x.shape
    assert h % window_size == 0
    assert w % window_size == 0
    nh, nw = h // window_size, w // window_size
    x = ut.reshape(x, [-1, nh, window_size, nw, window_size, c])
    x = ut.transpose(x, [0, 1, 3, 2, 4, 5])
    x = ut.reshape(x, [-1, window_size, window_size, c])
    return x, nh, nw


def window_reverse(x, nh, nw):
    _, window_size, c = x.shape
    window_size = int(np.sqrt(window_size))
    x = tf.reshape(x, [-1, nh, nw, window_size, window_size, c])
    x = tf.transpose(x, [0, 1, 3, 2, 4, 5])
    x = tf.reshape(x, [-1, nh * window_size, nw * window_size, c])
    return x


class WindowAttention(kr.Model, ABC):
    def __init__(self, num_heads, dropout=.1):
        super(WindowAttention, self).__init__()
        self.qkv = None
        self.scale = None
        self.proj = None
        self.num_heads = num_heads
        self.dropout = kr.layers.Dropout(dropout)

    def call(self, inputs, training=None, mask=None):
        _, n, c = inputs.shape
        assert c % self.num_heads == 0
        n_dim = c // self.num_heads
        self.qkv = self.qkv or kr.layers.Dense(3 * c)
        self.scale = self.scale or np.sqrt(n_dim)
        qkv = self.qkv(inputs)
        qkv = tf.reshape(qkv, [-1, n, 3, self.num_heads, n_dim])
        qkv = tf.transpose(qkv, [0, 2, 3, 1, 4])
        q, k, v = tf.unstack(qkv, axis=1)
        kT = tf.transpose(k, [0, 1, 3, 2])
        score = q @ kT
        score = score / self.scale
        if mask is not None:
            nw = mask.shape[0]
            score = tf.reshape(score, [-1, nw, self.num_heads, n, n])
            mask = tf.expand_dims(mask, 1)
            mask = tf.expand_dims(mask, 0)
            score = score + mask
            score = tf.reshape(score, [-1, self.num_heads, n, n])
        score = tf.nn.softmax(score, -1)
        attn = score @ v
        attn = self.dropout(attn)
        attn = tf.transpose(attn, [0, 2, 1, 3])
        attn = tf.reshape(attn, [-1, n, c])
        self.proj = self.proj or kr.layers.Dense(c)
        x = self.proj(attn)
        x = self.dropout(x)
        return x


def attention_mask(h, w, window_size, shift_size):
    img_mask = np.zeros([1, h, w, 1])
    h_slices = (slice(0, -window_size),
                slice(-window_size, -shift_size),
                slice(-shift_size, None))
    cnt = 0
    for h in h_slices:
        for w in h_slices:
            img_mask[:, h, w, :] = cnt
            cnt += 1
    mask_windows, nh, nw = window_partition(img_mask, window_size, ut=np)
    mask_windows = np.reshape(mask_windows, [-1, window_size ** 2])
    attn_mask = np.expand_dims(mask_windows, 1) - np.expand_dims(mask_windows, 2)
    attn_mask = np.where(attn_mask != 0, -100., 0.)
    return attn_mask.astype(np.float32)


class TransformerBlock(kr.Model, ABC):
    def __init__(self, window_size=4, num_heads=8,
                 normalize=kr.layers.LayerNormalization, dropout=.2):
        super(TransformerBlock, self).__init__()
        self.norms = [normalize() for _ in range(4)]
        self.wmsa = [WindowAttention(num_heads, dropout) for _ in range(2)]
        self.mlp = None
        self.dropout = dropout
        self.window_size = window_size
        self.shift_size = self.window_size // 2
        self.mask = None

    def call(self, inputs, training=None, mask=None):
        _, h, w, c = inputs.shape
        if self.mask is None:
            self.mask = attention_mask(h, w, self.window_size, self.shift_size)
        self.mlp = self.mlp or [Mlp(c, 4 * c, self.dropout) for _ in range(2)]
        shortcut = x = inputs
        x = self.norms[0](x)
        x, nh, nw = window_partition(x, self.window_size)
        x = tf.reshape(x, [-1, self.window_size ** 2, c])
        x = self.wmsa[0](x)
        x = window_reverse(x, nh, nw)
        shortcut = x = x + shortcut
        x = self.norms[1](x)
        x = self.mlp[0](x)
        shortcut = x = x + shortcut
        x = self.norms[2](x)
        x = tf.roll(x, (-self.shift_size, -self.shift_size), (1, 2))
        x, nh, nw = window_partition(x, self.window_size)
        x = tf.reshape(x, [-1, self.window_size ** 2, c])
        x = self.wmsa[1](x, mask=self.mask)
        x = window_reverse(x, nh, nw)
        x = tf.roll(x, (self.shift_size, self.shift_size), (1, 2))
        shortcut = x = x + shortcut
        x = self.norms[3](x)
        x = self.mlp[1](x)
        x += shortcut
        return x


class Mlp(kr.Model, ABC):
    def __init__(self, input_dim, hidden_dim, dropout=0.):
        super(Mlp, self).__init__()
        self.dense1 = kr.layers.Dense(hidden_dim, kr.activations.gelu)
        self.dropout = kr.layers.Dropout(dropout)
        self.dense2 = kr.layers.Dense(input_dim)

    def call(self, inputs, training=None, mask=None):
        x = self.dense1(inputs)
        x = self.dropout(x)
        x = self.dense2(x)
        x = self.dropout(x)
        return x


class Transformer(kr.Model, ABC):
    def __init__(self, layers, embed_dim=96, patch_size=4, normalize=kr.layers.LayerNormalization,
                 window_size=8, num_heads=8, dropout=.2):
        super(Transformer, self).__init__()
        self.patchEmbed = PatchEmbed(embed_dim, patch_size, normalize)
        self.block = [kr.Sequential([TransformerBlock(window_size, num_heads, normalize, dropout)
                                     for _ in range(layer)]) for layer in layers]
        self.patchMerging = [PatchMerging(normalize) for _ in range(1, len(layers))]

    def call(self, inputs, training=None, mask=None):
        x = self.patchEmbed(inputs)
        for block, merging in zip(self.block, self.patchMerging):
            x = block(x)
            x = merging(x)
        x = self.block[-1](x)
        return x


if __name__ == '__main__':
    '''
    Patch Embedding.
    '''
    # x_train = np.random.uniform(size=[3, 128, 128, 3])
    # patch_embed = PatchEmbed()
    # x_result = patch_embed(x_train)
    # print(x_result, x_result.shape)
    '''
    Patch Merging.
    '''
    # x_train = np.random.uniform(size=[3, 128, 128, 3])
    # patch_merging = PatchMerging()
    # x_result = patch_merging(x_train)
    # print(x_result, x_result.shape)
    '''
    window partition reverse.
    '''
    # x_train = np.random.uniform(size=[3, 64, 64, 48])
    # x_result, nh, nw = window_partition(x_train, 4)
    # print(x_result.shape)
    # x_result = window_reverse(x_result, nh, nw)
    # print(x_result.shape)
    '''
    window attention.
    '''
    # x_train = np.random.uniform(size=[768, 16, 96])
    # windowAttention = WindowAttention(8)
    # x_result = windowAttention(x_train)
    # print(x_result.shape)
    '''
    attention mask.
    '''
    # attn_mask = attention_mask(8, 8, 4, 2)
    # print(attn_mask)
    '''
    shifted window attention.
    '''
    # x_train = np.random.uniform(size=(3, 32, 32, 96))
    # x_train, _, _ = window_partition(x_train, 4)
    # x_train = tf.reshape(x_train, [-1, 16, 96])
    # attn_mask = attention_mask(32, 32, 4, 2)
    # windowAttention = WindowAttention(8)
    # x_result = windowAttention(x_train, mask=attn_mask)
    # print(x_result)
    '''
    transformer block.
    '''
    # x_train = np.random.uniform(size=[17, 32, 32, 96])
    # block = TransformerBlock()
    # x_result = block(x_train)
    # print(x_result)
    '''
    swin transformer.
    '''
    x_train = np.random.uniform(size=[3, 128, 128, 96])
    # 32, 16, 8
    transformer = Transformer([2, 3, 1])
    x_result = transformer(x_train)
    print(x_result)
