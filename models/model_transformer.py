from abc import ABC
import tensorflow.keras as kr
import tensorflow as tf
from tensorflow.keras.datasets import cifar10
from horse2zebra.horse2zebra_datasets12 import StandardScaler


class SwinTranformer(kr.Model, ABC):
    def __init__(self, img_size=128, channel=3):
        super(SwinTranformer, self).__init__()
        self.conv = kr.layers.Conv2D(96, 2, 2)
        # 16 16 32
        self.block = SwinTranformerBlock(16, win_size=8, multi_header=3, units=32)
        # 8 8 64
        self.merging = Merging(3 * 64)
        # 8 8 64
        self.block2 = SwinTranformerBlock(8, win_size=8, multi_header=3, units=64)
        # 4 4 128
        self.merging2 = Merging(3 * 128)
        # 4 4 128
        self.block3 = SwinTranformerBlock(4, win_size=4, multi_header=3, units=128)
        self.build([None, img_size, img_size, channel])

    def call(self, inputs, training=None, mask=None):
        x = self.conv(inputs)
        x = self.block(x)
        x = self.merging(x)
        x = self.block2(x)
        x = self.merging2(x)
        x = self.block3(x)
        return x


class Merging(kr.Model, ABC):
    def __init__(self, units):
        super(Merging, self).__init__()
        self.ln = kr.layers.LayerNormalization(epsilon=1e-5)
        self.dense = kr.layers.Dense(units)

    def call(self, inputs, training=None, mask=None):
        x0 = inputs[:, ::2, ::2, :]
        x1 = inputs[:, ::2, 1::2, :]
        x2 = inputs[:, 1::2, ::2, :]
        x3 = inputs[:, 1::2, 1::2, :]
        x = tf.concat([x0, x1, x2, x3], -1)
        x = self.ln(x)
        x = tf.nn.tanh(x)
        x = self.dense(x)
        return x


class SwinTranformerBlock(kr.Model, ABC):
    def __init__(self, img_size, win_size=8, multi_header=3, units=32):
        super(SwinTranformerBlock, self).__init__()
        assert img_size % win_size == 0, 'win_size error!'
        self.ln = kr.layers.LayerNormalization(epsilon=1e-5)
        self.qkv = kr.layers.Dense(3 * units)
        self.sh = img_size // win_size
        self.pos = kr.layers.Embedding(self.sh ** 2, units)
        self.dense = kr.layers.Dense(units, activation='tanh')
        self.win_size, self.multi, self.units = win_size, multi_header, units

    def call(self, inputs, training=None, mask=None):
        x = self.ln(inputs)
        x = tf.reshape(x, [-1, self.sh, self.win_size, self.sh, self.win_size, self.multi, self.units])
        x = tf.transpose(x, [0, 5, 1, 3, 2, 4, 6])
        # 3 16 64 32
        x = tf.reshape(x, [-1, self.multi, self.sh ** 2, self.win_size ** 2, self.units])
        # 3 16 64 3 * 32
        qkv = self.qkv(x)
        # 3 16 64 32
        q, k, v = tf.split(qkv, 3, -1)
        qkt = tf.einsum('...msfc,...mskc->...msfk', q, k)
        # 3 16 64 64
        qkt = tf.nn.softmax(qkt / tf.sqrt(float(self.units)))
        # 3 16 64 32
        x = qkt @ v
        bis = [self.pos(float(i)) for i in range(self.sh ** 2)]
        bis = tf.stack(bis)
        x = tf.einsum('...msfc,sc->...msfc', x, bis)
        x = tf.reshape(x, [-1, self.multi, self.sh, self.sh, self.win_size, self.win_size, self.units])
        x = tf.transpose(x, [0, 2, 4, 3, 5, 1, 6])
        rs = self.sh * self.win_size
        x = tf.reshape(x, [-1, rs, rs, self.multi * self.units])
        x = self.dense(x)
        return x


if __name__ == '__main__':
    scaler = StandardScaler()
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    x_train, x_test = scaler.fit_transform(x_train), scaler.fit_transform(x_test)
    model = SwinTranformer(img_size=32)
    model.summary()
    model = kr.Sequential([
        model,
        kr.layers.Flatten(),
        kr.layers.Dense(10)
    ])
    model.compile(optimizer=kr.optimizers.Adam(),
                  loss=kr.losses.SparseCategoricalCrossentropy(True),
                  metrics=[kr.metrics.sparse_categorical_accuracy])
    model.fit(x_train, y_train, 32, 5, validation_data=(x_test, y_test))

