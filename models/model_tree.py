import tensorflow.keras as kr
import tensorflow as tf
from abc import ABC
from tensorflow.keras.datasets import cifar10
from horse2zebra.horse2zebra_datasets12 import StandardScaler
import matplotlib.pyplot as plt


def conv_init(units, kernel_size):
    down = kr.Sequential([
        kr.layers.Conv2D(units, kernel_size),
        kr.layers.LayerNormalization(epsilon=1e-5),
        kr.layers.LeakyReLU(),
        kr.layers.MaxPool2D(),
        kr.layers.Dropout(.2)
    ])
    return down


class Tree(kr.Model, ABC):
    def __init__(self):
        super(Tree, self).__init__()
        # 16
        self.conv1 = conv_init(32, 1)
        # 7
        self.conv2 = [conv_init(64, 3) for _ in range(2)]
        # 1
        self.conv3 = [conv_init(96, 5) for _ in range(4)]
        self.dense = self.add_weight(shape=[4, 96, 128], dtype=self.dtype,
                                     initializer='glorot_uniform', trainable=True)

    def call(self, inputs, training=None, mask=None):
        x = self.conv1(inputs)
        x = [conv(x) for conv in self.conv2]
        x = [conv(x[i // 2]) for i, conv in enumerate(self.conv3)]
        x = tf.stack(x, -1)
        x = tf.einsum('...hwcs,sco->...hwo', x, self.dense)
        return x


class DownUp(kr.Model, ABC):
    def __init__(self, fetures, channel=3, callback=None):
        super(DownUp, self).__init__()
        downs, ups = [], []
        for i, (units, kernel_size) in enumerate(fetures):
            conv = kr.layers.Conv2D(units, kernel_size)
            pool = kr.layers.MaxPool2D()
            conv = kr.Sequential([conv, pool]) if not callable(callback) else callback(conv, pool)
            downs.append(conv)
            pool = kr.layers.UpSampling2D()
            conv = kr.layers.Conv2DTranspose(channel, kernel_size)
            channel = units
            if not callable(callback) or i == 0:
                conv = kr.Sequential([pool, conv])
            else:
                conv = callback(pool, conv)
            ups.append(conv)
        self.downs, self.ups = downs, ups[::-1]

    def call(self, inputs, training=None, mask=None):
        x = inputs
        for down in self.downs:
            x = down(x)
        for up in self.ups:
            x = up(x)
        return x


class Down(kr.Model, ABC):
    def __init__(self, fetures, callback=None):
        super(Down, self).__init__()
        downs = []
        for units, kernel_size in fetures:
            conv = kr.layers.Conv2D(units, kernel_size)
            pool = kr.layers.MaxPool2D()
            conv = kr.Sequential([conv, pool]) if not callable(callback) else callback(conv, pool)
            downs.append(conv)
        self.downs = downs

    def call(self, inputs, training=None, mask=None):
        x = inputs
        for down in self.downs:
            x = down(x)
        return x


class ConvLSTM(kr.Model, ABC):
    def __init__(self, block, channel=3, sequence=9, recycle=3, kernel_size=1):
        super(ConvLSTM, self).__init__()
        self.blocks = [block() for _ in range(sequence)]
        self.lstm = kr.Sequential([
            kr.layers.ConvLSTM2D(channel, kernel_size, return_sequences=i != recycle - 1) for i in range(recycle)
        ])

    def call(self, inputs, training=None, mask=None):
        x = inputs
        xs = []
        for block in self.blocks:
            x = block(x)
            xs.append(x)
        x = tf.stack(xs, 1)
        x = self.lstm(x)
        return x


class PercentagesConv(kr.Model, ABC):
    def __init__(self, block, seq=9, percentage=.9):
        super(PercentagesConv, self).__init__()
        self.blocks = [block() for _ in range(seq)]
        self.percentages = [percentage ** (seq - i) for i in range(seq)]

    def call(self, inputs, training=None, mask=None):
        x = inputs
        xs = []
        for block in self.blocks:
            x = block(x)
            xs.append(x)
        x = [x * percentage for x, percentage in zip(xs, self.percentages)]
        x = tf.reduce_sum(x, axis=0)
        return x


class ConvLSTMD(kr.Model, ABC):
    def __init__(self, block, channel=3, sequence=9, recycle=3, kernel_size=1):
        super(ConvLSTMD, self).__init__()
        self.blocks = [block() for _ in range(sequence)]
        self.lstm = kr.Sequential([
            kr.layers.ConvLSTM2D(channel, kernel_size, return_sequences=i != recycle - 1) for i in range(recycle)
        ])

    def call(self, inputs, training=None, mask=None):
        x = inputs
        xs = [block(x) for block in self.blocks]
        x = tf.stack(xs, 1)
        x = self.lstm(x)
        return x


if __name__ == '__main__':
    scaler = StandardScaler()
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    x_train, x_test = scaler.fit_transform(x_train), scaler.fit_transform(x_test)
    callback = lambda conv, pool: kr.Sequential([
        conv,
        kr.layers.LayerNormalization(),
        kr.layers.LeakyReLU(),
        pool,
        kr.layers.Dropout(.2)
    ])
    # model = ConvLSTM(lambda: DownUp([(48, 1),
    #                                  (96, 3),
    #                                  (192, 4)], callback=callback))
    model = PercentagesConv(lambda: DownUp([(48, 1),
                                            (96, 3),
                                            (192, 4)], callback=callback))
    xp = model(x_train[-1:])
    plt.subplot(121)
    plt.imshow(x_train[-1])
    plt.subplot(122)
    plt.imshow(xp[0])
    plt.show()
    # model = kr.Sequential([
    #     ConvLSTM(lambda: DownUp([(24, 1),
    #                              (48, 3)], callback=callback),
    #              kernel_size=3, sequence=3, recycle=2),
    #     kr.layers.Flatten(),
    #     kr.layers.Dense(10)
    # ])
    model = kr.Sequential([
        Tree(),
        kr.layers.Flatten(),
        kr.layers.Dense(10)
    ])
    model.compile(optimizer=kr.optimizers.Adam(),
                  loss=kr.losses.SparseCategoricalCrossentropy(True),
                  metrics=[kr.metrics.sparse_categorical_accuracy])
    model.fit(x_train, y_train, 32, 5, validation_data=(x_test, y_test))
    model.summary()
