import tensorflow.keras as kr
import tensorflow as tf
from tensorflow.keras.datasets import cifar10
from trm.gru_transformer import StandardScaler
import matplotlib.pyplot as plt


class RollingConv(kr.Model):
    def __init__(self, units, kernel_size, rolling=2, strides=2, padding='same',
                 conv=kr.layers.Conv2D, callback=None):
        super(RollingConv, self).__init__()
        self.convs = [conv(units, kernel_size, strides, padding) for _ in range(rolling)]
        if callable(callback):
            self.convs = [callback(conv) for conv in self.convs]
        self.dense = kr.layers.Dense(1)

    def call(self, inputs, training=None, mask=None):
        x = [conv(inputs) for conv in self.convs]
        x = tf.stack(x, -1)
        x = self.dense(x)
        x = tf.squeeze(x, -1)
        return x


class UNet(kr.Model):
    def __init__(self, layers, channel=3):
        """
        :param layers: units, kernel_size, rolling=2, strides=2, callback=None
        :param channel: 图片通道数
        :param callback: models.model_skip.default_layer 包装函数
        """
        super(UNet, self).__init__()
        self.downs, self.ups, n = [], [], len(layers)
        assert n > 1, 'Error: less 2 layers!'
        for i in range(n):
            units, kernel_size, rolling, strides, callback = layers[i]
            down = RollingConv(units, kernel_size, rolling, strides, 'same', callback=callback)
            self.downs.append(down)
            units, kernel_size, rolling, strides, callback = layers[-1 - i]
            if i == n - 1:
                break
            units = layers[-2 - i][0]
            up = RollingConv(units, kernel_size, rolling, strides, 'same',
                             conv=kr.layers.Conv2DTranspose, callback=callback)
            dense = kr.layers.Dense(1)
            self.ups.append((up, dense))
        self.last = RollingConv(channel, kernel_size, rolling, strides, 'same', conv=kr.layers.Conv2DTranspose)

    def call(self, inputs, training=None, mask=None):
        x, xs = inputs, []
        for down in self.downs:
            x = down(x)
            xs.append(x)
        for (up, dense), dx in zip(self.ups, xs[::-1][1:]):
            x = up(x)
            x = tf.stack([x, dx], -1)
            x = dense(x)
            x = tf.squeeze(x, -1)
        x = self.last(x)
        return x


if __name__ == '__main__':
    scaler = StandardScaler()
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    x_train, x_test = scaler.fit_transform(x_train), scaler.transform(x_test)
    callback = lambda conv: kr.Sequential([
        conv,
        kr.layers.LayerNormalization(epsilon=1e-5),
        kr.layers.LeakyReLU()
    ])
    callback2 = lambda conv: kr.Sequential([
        callback(conv),
        kr.layers.MaxPooling2D(3, 1, 'same')
    ])

    model = UNet([(12, 7, 5, 2, callback2),
                  (24, 5, 3, 2, callback),
                  (24, 5, 2, 2, callback),
                  (48, 3, 1, 2, callback),
                  (48, 3, 1, 2, callback2)])
    xp = model(x_train[-1:])
    model.summary()
    plt.subplot(121)
    plt.imshow(xp[0])
    plt.subplot(122)
    plt.imshow(x_train[-1])
    plt.show()

    model = kr.Sequential([
        RollingConv(12, 7, 5, callback=callback2),
        RollingConv(24, 5, 3, callback=callback),
        RollingConv(24, 5, 2, callback=callback),
        RollingConv(48, 3, 1, callback=callback2),
        RollingConv(48, 3, 1, callback=callback2),
        kr.layers.Flatten(),
        kr.layers.Dense(10)
    ])
    model.compile(optimizer=kr.optimizers.Adam(),
                  loss=kr.losses.SparseCategoricalCrossentropy(True),
                  metrics=[kr.metrics.sparse_categorical_accuracy])
    model.fit(x_train, y_train, 64, 5, validation_data=(x_test, y_test))
    model.summary()
