import tensorflow.keras as kr
import tensorflow as tf
from tensorflow.keras.datasets import cifar10
from horse2zebra.horse2zebra_datasets12 import StandardScaler


class SwinTransformer(kr.Model):
    def __init__(self, c, blocks=[2, 2, 4, 2], patch_size=4, channel=3, window_size=4, layer=None):
        super(SwinTransformer, self).__init__()
        self.embedding = PatchEmbedding(c, patch_size, channel)
        self.base_layers = [([SwinTransformerBlock(c * 2 ** i, window_size) for _ in range(block)],
                             (layer() if layer is not None else None) if i != len(blocks) - 1 else None,
                             PatchMerging(c * 2 ** i * 2) if i != len(blocks) - 1 else None)
                            for i, block in enumerate(blocks)]
        self.roll = window_size // 2

    def call(self, inputs, training=None, mask=None):
        x = self.embedding(inputs)
        for blocks, layer, merging in self.base_layers:
            for i, block in enumerate(blocks):
                bx = tf.roll(x, [-self.roll, -self.roll], axis=[1, 2]) if i % 2 == 1 else x
                bx = block(bx)
                bx = bx if i % 2 != 1 else tf.roll(bx, [self.roll, self.roll], [1, 2])
                x = tf.reduce_max([x, bx], 0)
            x = merging(x) if merging is not None else x
            x = layer(x) if layer is not None else x
        return x


class SwinTransformerTranspose(kr.Model):
    def __init__(self, c, blocks=[2, 4, 2, 2], patch_size=4, channel=3, window_size=4, layer=None):
        super(SwinTransformerTranspose, self).__init__()
        n = len(blocks)
        self.base_layers = [([SwinTransformerBlock(c * 2 ** (n - i - 1), window_size) for _ in range(block)],
                             (layer() if layer is not None else None) if i != 0 else None,
                             PatchMergingTranspose(c * 2 ** (n - i - 1)) if i != 0 else None)
                            for i, block in enumerate(blocks)]
        self.embedding = PatchEmbeddingTranspose(c, patch_size, channel)
        self.roll = window_size // 2

    def call(self, inputs, training=None, mask=None):
        x = inputs
        for blocks, layer, merging in self.base_layers:
            x = merging(x) if merging is not None else x
            for i, block in enumerate(blocks):
                bx = tf.roll(x, [-self.roll, -self.roll], axis=[1, 2]) if i % 2 == 1 else x
                bx = block(bx)
                bx = bx if i % 2 != 1 else tf.roll(bx, [self.roll, self.roll], [1, 2])
                x = tf.reduce_max([x, bx], 0)
            x = layer(x) if layer is not None else x
        x = self.embedding(x)
        return x


class SwinTransformerBlock(kr.Model):
    def __init__(self, units, window_size=4):
        super(SwinTransformerBlock, self).__init__()
        self.qkv = self.add_weight('qkv', (3, units, units), dtype=self.dtype,
                                   initializer='glorot_uniform', trainable=True)
        self.window_size, self.units = window_size, units

    def call(self, inputs, training=None, mask=None):
        _, h, w, c = inputs.shape
        assert h % self.window_size == 0 and w % self.window_size == 0, 'window size mismatch!'
        hn, wn = h // self.window_size, w // self.window_size
        x = tf.reshape(inputs, [-1, hn, self.window_size, wn, self.window_size, c])
        x = tf.transpose(x, [0, 1, 3, 2, 4, 5])
        x = tf.reshape(x, [-1, self.window_size ** 2, c])
        qkv = tf.einsum('...wc,nco->...nwo', x, self.qkv)
        q, k, v = tf.unstack(qkv, axis=1)
        q = tf.einsum('...wc,...ic->...wi', q, k)
        q = tf.nn.softmax(q, -1)
        x = q @ v
        x = tf.reshape(x, [-1, hn, wn, self.window_size, self.window_size, self.units])
        x = tf.transpose(x, [0, 1, 3, 2, 4, 5])
        x = tf.reshape(x, [-1, h, w, self.units])
        return x


class UNet(kr.Model):
    def __init__(self, c, blocks=[2, 2, 4, 2], patch_size=4, channel=3, window_size=4, layer=None):
        super(UNet, self).__init__()
        self.down = SwinTransformer(c, blocks, patch_size, channel, window_size, layer)
        self.up = SwinTransformerTranspose(c, blocks[::-1], patch_size, channel, window_size, layer)

    def call(self, inputs, training=None, mask=None):
        x = self.down(inputs)
        x = self.up(x)
        return x


class PatchEmbedding(kr.Model):
    def __init__(self, c, patch_size=4, channel=3):
        super(PatchEmbedding, self).__init__()
        self.conv = kr.layers.Conv2D(patch_size ** 2 * channel, patch_size, patch_size)
        self.embedding = kr.layers.Dense(c)

    def call(self, inputs, training=None, mask=None):
        x = self.conv(inputs)
        x = self.embedding(x)
        return x


class PatchEmbeddingTranspose(kr.Model):
    def __init__(self, c, patch_size=4, channel=3):
        super(PatchEmbeddingTranspose, self).__init__()
        self.conv = kr.layers.Conv2DTranspose(channel, patch_size, patch_size)
        self.embedding = kr.layers.Dense(c)

    def call(self, inputs, training=None, mask=None):
        x = self.embedding(inputs)
        x = self.conv(x)
        return x


# class PatchMergingTranspose(kr.Model):
#     def __init__(self, c):
#         super(PatchMergingTranspose, self).__init__()
#         self.conv = kr.layers.Conv2DTranspose(c, 2, 2)
#
#     def call(self, inputs, training=None, mask=None):
#         x = self.conv(inputs)
#         return x

class PatchMergingTranspose(kr.Model):
    def __init__(self, c):
        super(PatchMergingTranspose, self).__init__()
        self.conv = kr.layers.UpSampling2D()
        self.conv2 = kr.layers.Conv2D(c, 1)

    def call(self, inputs, training=None, mask=None):
        x = self.conv(inputs)
        x = self.conv2(x)
        return x


# class PatchMerging(kr.Model):
#     def __init__(self, c):
#         super(PatchMerging, self).__init__()
#         self.conv = kr.layers.Conv2D(c, 2, 2)
#
#     def call(self, inputs, training=None, mask=None):
#         x = self.conv(inputs)
#         return x


class PatchMerging(kr.Model):
    def __init__(self, c):
        super(PatchMerging, self).__init__()
        self.dense = self.add_weight('dmerge', (2 * c, c), dtype=self.dtype,
                                     initializer=kr.initializers.GlorotUniform(),
                                     trainable=True)

    def call(self, inputs, training=None, mask=None):
        x1 = inputs[:, ::2, ::2, :]
        x2 = inputs[:, ::2, 1::2, :]
        x3 = inputs[:, 1::2, ::2, :]
        x4 = inputs[:, 1::2, 1::2, :]
        x = tf.concat([x1, x2, x3, x4], -1)
        x = tf.einsum('...hwc,co->...hwo', x, self.dense)
        return x


if __name__ == '__main__':
    scaler = StandardScaler()
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    x_train, x_test = scaler.fit_transform(x_train), scaler.fit_transform(x_test)

    model = PatchEmbedding(48)
    px = model(x_train[-1:])
    model.summary()
    print(px.shape)

    model = SwinTransformerBlock(48)
    px = model(px)
    model.summary()
    print(px.shape)

    model = PatchMerging(96)
    px = model(px)
    model.summary()
    print(px.shape)

    # 8 4
    model = UNet(96, blocks=[2, 2], layer=lambda: kr.Sequential([
        kr.layers.LayerNormalization(epsilon=1e-5),
        kr.layers.ReLU()]))
    px = model(x_train[-1:])
    model.summary()
    print(px.shape)

    model = SwinTransformer(24, patch_size=2, window_size=2, layer=lambda: kr.Sequential([
        kr.layers.LayerNormalization(epsilon=1e-5),
        kr.layers.ReLU()]))
    px = model(x_train[-1:])
    model.summary()
    print(px.shape)

    model = kr.Sequential([
        model, kr.layers.Flatten(), kr.layers.Dense(10)
    ])
    model.compile(optimizer=kr.optimizers.Adam(),
                  loss=kr.losses.SparseCategoricalCrossentropy(True),
                  metrics=[kr.metrics.sparse_categorical_accuracy])
    model.fit(x_train, y_train, 64, 5, validation_data=(x_test, y_test))
    model.summary()
