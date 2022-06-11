import tensorflow.keras as kr
import tensorflow as tf
from tensorflow.keras.datasets import cifar10
from trm.gru_transformer import StandardScaler
import matplotlib.pyplot as plt


class ConvTransformer(kr.Model):
    def __init__(self, convs, splits=2,
                 pooling=kr.layers.MaxPooling2D(2, 1, 'same')):
        super(ConvTransformer, self).__init__()
        self.convs, self.splits, self.pooling = convs, splits, pooling
        self.dense = kr.layers.Dense(1)

    def call(self, inputs, training=None, mask=None):
        hs = tf.split(inputs, self.splits, 1)
        xh, xs = [], [self.convs[i](inputs) for i in range(self.splits)]
        for h in hs:
            ws = tf.split(h, self.splits, 2)
            xw = []
            for i in range(self.splits):
                x = self.convs[i](ws[i])
                xw.append(x)
            xw = tf.concat(xw, 2)
            xh.append(xw)
        x = tf.concat(xh, 1)
        xs.append(x)
        x = tf.stack(xs, -1)
        x = self.dense(x)
        x = tf.squeeze(x, -1)
        x = self.pooling(x)
        return x


def create_skips(layers, skip=2, splits=2, start_activation=2,
                 pooling=kr.layers.MaxPooling2D(2, 1, 'same'),
                 con=kr.layers.Conv2D, activation=tf.nn.leaky_relu):
    ds, i, strides_, n = [], 0, 1, len(layers)
    for units, kernel_size, strides in layers:
        i += 1
        activation_ = None if i < start_activation or i == n else activation
        conv = ConvTransformer([con(units, kernel_size, strides, 'same', activation=activation_)
                                for _ in range(splits)], splits, pooling)
        if i % skip == 0:
            conv_ = ConvTransformer([con(units, kernel_size, strides_ * strides, 'same')
                                     for _ in range(splits)], splits, pooling)
            ds.append((conv, conv_))
            strides_ = 1
        else:
            strides_ *= strides
            ds.append(conv)
    return ds


class SkipModel(kr.Model):
    def __init__(self, skips, snorm=2, norm=kr.layers.LayerNormalization, epsilon=1e-5):
        """
        model = kr.Sequential([
            SkipModel(create_skips([(12, 5, 2), (48, 3, 2), (96, 3, 2)])),
            SkipModel(create_skips([(48, 3, 2), (12, 3, 2), (3, 5, 2)], con=kr.layers.Conv2DTranspose))
        ])
        :param skips: create_skips([(48, 3, 2), (12, 3, 2), (3, 5, 2)], con=kr.layers.Conv2DTranspose)
        :param snorm:
        :param norm:
        :param epsilon:
        """
        super(SkipModel, self).__init__()
        self.down_skips = skips
        self.norm, self.snorm = norm(epsilon=epsilon), snorm

    def call(self, inputs, training=None, mask=None):
        x, i, n = inputs, 0, len(self.down_skips)
        for down_skip in self.down_skips:
            i += 1
            if not hasattr(down_skip, '__len__'):
                x = down_skip(x)
            else:
                x = down_skip[0](x)
                x_ = down_skip[1](inputs)
                x = tf.concat([x, x_], -1)
            if i > self.snorm and i != n:
                x = self.norm(x)
        return x


if __name__ == '__main__':
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    scaler = StandardScaler()
    x_train, x_test = scaler.fit_transform(x_train), scaler.transform(x_test)
    conv = kr.layers.Conv2DTranspose(3, 3, 2, 'same')

    # conv1 = kr.layers.Conv2DTranspose(3, 3, 2, 'same')
    # conv2 = kr.layers.Conv2D(12, 5, 2, 'same')
    # mode1 = kr.Sequential([conv, kr.layers.MaxPooling2D(2, 1, 'same')])
    # model = kr.Sequential([ConvTransformer([conv2, conv2], pooling=kr.layers.MaxPooling2D(2, 1, 'same')),
    #                        ConvTransformer([conv1, conv1], pooling=kr.layers.MaxPooling2D(2, 1, 'same'))])

    model = kr.Sequential([
        SkipModel(create_skips([(12, 5, 2), (48, 3, 2), (96, 3, 2)])),
        SkipModel(create_skips([(48, 3, 2), (12, 3, 2), (3, 5, 2)], con=kr.layers.Conv2DTranspose))
    ])

    xc = conv(x_train[-1:])
    xp = model(x_train[-1:])
    plt.subplot(131)
    plt.imshow(xc[0])
    plt.subplot(132)
    plt.imshow(xp[0])
    plt.subplot(133)
    plt.imshow(x_train[-1])
    plt.show()

    # model = kr.Sequential([
    #     SkipModel(create_skips([(12, 5, 2), (48, 3, 2), (96, 3, 2)])),
    #     kr.layers.Flatten(),
    #     kr.layers.Dense(10)
    # ])
    # model.compile(optimizer=kr.optimizers.Adam(),
    #               loss=kr.losses.SparseCategoricalCrossentropy(True),
    #               metrics=[kr.metrics.sparse_categorical_accuracy])
    # model.fit(x_train, y_train, 64, 5, validation_data=(x_test, y_test))

    model.summary()
