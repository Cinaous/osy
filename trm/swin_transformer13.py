import math
from abc import ABC
import numpy as np
import tensorflow.keras as kr
import tensorflow as tf
from tensorflow.keras.datasets import mnist, cifar10
import matplotlib.pyplot as plt


class SwinUNet(kr.Model, ABC):
    def __init__(self, layers, output_dim, out_channel=3,
                 activation=kr.activations.relu, rate=.2):
        """
        :param layers: [(header, wind_size), ...]
        :param output_dim:
        :param activation:
        """
        super(SwinUNet, self).__init__()
        self.swins, n, self.downs, self.ups, self.swinus = [], 0, [], [], []
        for header, window_size in layers:
            n_dim = 2 ** n * output_dim
            swin = [MLP(n_dim, window_size, i != 0) for i in range(header)]
            self.swins.append(swin)
            down = kr.Sequential([
                PatchMegre(2 * n_dim, activation),
                kr.layers.LayerNormalization(epsilon=1e-5)
            ])
            n += 1
            self.downs.append(down)
            if n != len(layers):
                up = kr.Sequential([
                    kr.layers.Conv2DTranspose(n_dim, 2, 2, 'same'),
                    kr.layers.LayerNormalization(epsilon=1e-5),
                    kr.layers.Activation(activation),
                    kr.layers.Dropout(rate)
                ])
                self.ups.append(up)
                swin = [MLP(n_dim, layers[n][1], i != 0) for i in range(header)]
                self.swinus.append(swin)
        self.ups, self.swinus = self.ups[::-1], self.swinus[::-1]
        self.last = kr.Sequential([
            kr.layers.Conv2DTranspose(out_channel, 2, 2, 'same'),
            MLP(out_channel, layers[0][1])
        ])

    def call(self, inputs, training=None, mask=None):
        x, downs = inputs, []
        for swin, down in zip(self.swins, self.downs):
            x = down(x)
            ix = None
            for mlp in swin:
                x = mlp(x)
                ix = x if ix is None else ix
            x = x + ix if len(swin) > 1 else x
            downs.append(x)
        for swinu, up, xd in zip(self.swinus, self.ups, downs[::-1][1:]):
            x = up(x)
            x += xd
            ix = None
            for mlp in swinu:
                x = mlp(x)
                ix = x if ix is None else ix
            x = x + ix if len(swinu) > 1 else x
        x = self.last(x)
        return x


class SwinTransformer(kr.Model, ABC):
    def __init__(self, output_dim, layers, in_channel=3,
                 n_classes=None, kernel_size=4, strides=4, activation=kr.activations.relu):
        super(SwinTransformer, self).__init__()
        input_dim = kernel_size ** 2 * in_channel
        self.embedding = PatchEmbedding(input_dim, kernel_size, strides)
        self.base_layers, self.merge_layers, n = [], [], 0
        for header, window_size in layers:
            base_layer = [MLP(2 ** n * output_dim, window_size, i != 0) for i in range(header)]
            merge_layer = kr.Sequential([PatchMegre(2 ** n * output_dim, activation),
                                         kr.layers.LayerNormalization(epsilon=1e-5)]) if n != 0 else None
            self.base_layers.append(base_layer)
            self.merge_layers.append(merge_layer)
            n += 1
        self.gru = kr.layers.GRU(n_classes, activation=None) if n_classes is not None else None

    def call(self, inputs, training=None, mask=None):
        x = self.embedding(inputs)
        for base_layer, merge_layer in zip(self.base_layers, self.merge_layers):
            x = merge_layer(x) if merge_layer is not None else x
            ix = None
            for mlp in base_layer:
                x = mlp(x)
                ix = x if ix is None else ix
            x = x + ix if len(base_layer) > 1 else x
        _, h, w, c = x.shape
        x = x if self.gru is None else self.gru(tf.reshape(x, [-1, h * w, c]))
        return x


class PatchMegre(kr.Model, ABC):
    def __init__(self, output_dim, activation=kr.activations.relu):
        super(PatchMegre, self).__init__()
        self.embedding = kr.layers.Dense(output_dim, activation=activation)

    def call(self, inputs, training=None, mask=None):
        x = inputs
        x0 = x[:, ::2, ::2, :]
        x1 = x[:, ::2, 1::2, :]
        x2 = x[:, 1::2, ::2, :]
        x3 = x[:, 1::2, 1::2, :]
        x = tf.concat([x0, x1, x2, x3], -1)
        x = self.embedding(x)
        return x


class PatchEmbedding(kr.Model, ABC):
    def __init__(self, input_dim, kernel_size=4, strides=4):
        super(PatchEmbedding, self).__init__()
        self.conv = kr.layers.Conv2D(input_dim, kernel_size, strides, 'same')
        self.norm = kr.layers.LayerNormalization(epsilon=1e-5)

    def call(self, inputs, training=None, mask=None):
        x = self.conv(inputs)
        x = self.norm(x)
        return x


class MLP(kr.Model, ABC):
    def __init__(self, output_dim, window_size=4, shift=False):
        super(MLP, self).__init__()
        self.q = kr.layers.Dense(output_dim, use_bias=False)
        self.k = kr.layers.Dense(output_dim, use_bias=False)
        self.v = kr.layers.Dense(output_dim, use_bias=False)
        self.window_size, self.output_dim, self.shift = window_size, output_dim, shift

    def attention(self, mask):
        _, mh, mw, c = mask.shape
        mask = tf.reshape(mask, [-1, mh * mw, c])
        q = self.q(mask)
        k = self.k(mask)
        v = self.v(mask)
        kt = tf.transpose(k, [0, 2, 1])
        mask = q @ kt
        qd = tf.sqrt(tf.reduce_sum(tf.square(q), -1))[..., tf.newaxis]
        kd = tf.sqrt(tf.reduce_sum(tf.square(k), -1))[..., tf.newaxis]
        kd = tf.transpose(kd, [0, 2, 1])
        d = qd @ kd
        mask = mask / d
        mask = tf.nn.softmax(mask, -1)
        mask = mask @ v
        mask = tf.reshape(mask, [-1, mh, mw, self.output_dim])
        return mask

    def call(self, inputs, training=None, mask=None):
        x = inputs
        _, h, w, _ = x.shape
        ws = self.window_size
        ws = ws if ws < h else h
        if self.shift:
            x = tf.roll(x, [ws // 2, ws // 2], [1, 2])
        assert h == w and h % ws == 0, 'Error: trainA != w or trainA % ws != 0'
        n, hm = h // ws, []
        for i in range(n):
            si = i * ws
            ei = si + ws
            wm = []
            for j in range(n):
                sj = j * ws
                ej = sj + ws
                mask = x[:, si:ei, sj:ej, :]
                mask = self.attention(mask)
                wm.append(mask)
            wm = tf.concat(wm, 2)
            hm.append(wm)
        x = tf.concat(hm, 1)
        return x


if __name__ == '__main__':
    # (x_train, y_train), (x_test, y_test) = mnist.load_data()
    # x_train, x_test = x_train[..., np.newaxis] / 127.5 - 1, x_test[..., np.newaxis] / 127.5 - 1
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    x_train, x_test = x_train / 127.5 - 1, x_test / 127.5 - 1
    model = kr.Sequential([
        # 16 8 4
        SwinUNet([(1, 4)], 12),
        # 8, 4
        # SwinTransformer(48, [(2, 4), (4, 4)], n_classes=10),
        # kr.layers.Flatten(),
        # kr.layers.Dense(10)
    ])
    x = model(x_train[:1])
    plt.subplot(121)
    plt.imshow(x[0])
    plt.subplot(122)
    plt.imshow(x_train[0])
    plt.show()
    # model.compile(optimizer=kr.optimizers.Adam(),
    #               loss=kr.losses.SparseCategoricalCrossentropy(from_logits=True),
    #               metrics=[kr.metrics.sparse_categorical_accuracy])
    # model.fit(x_train, y_train, 64, 5, validation_data=(x_test, y_test), validation_freq=1)
    model.summary()
    model.layers[0].summary()
