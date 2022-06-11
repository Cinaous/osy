import tensorflow.keras as kr
import tensorflow as tf
from tensorflow.keras.datasets import cifar10
# from gru_transformer import StandardScaler
import matplotlib.pyplot as plt


class ConvTransformer(kr.Model):
    def __init__(self, convs, splits=2,
                 pooling=kr.layers.MaxPooling2D(2, 1, 'same')):
        super(ConvTransformer, self).__init__()
        self.convs, self.splits, self.pooling = convs, splits, pooling

    def call(self, inputs, training=None, mask=None):
        hs = tf.split(inputs, self.splits, 1)
        xh = []
        for h in hs:
            ws = tf.split(h, self.splits, 2)
            xw = []
            for i in range(self.splits):
                x = self.convs[i](ws[i])
                xw.append(x)
            xw = tf.concat(xw, 2)
            xh.append(xw)
        x = tf.concat(xh, 1)
        x = self.pooling(x)
        return x


if __name__ == '__main__':
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    # scaler = StandardScaler()
    # x_train, x_test = scaler.fit_transform(x_train), scaler.transform(x_test)
    conv = kr.layers.Conv2DTranspose(3, 3, 2, 'same')
    conv1 = kr.layers.Conv2DTranspose(3, 3, 2, 'same')
    conv2 = kr.layers.Conv2D(12, 5, 2, 'same')
    mode1 = kr.Sequential([conv, kr.layers.MaxPooling2D(2, 1, 'same')])
    model = kr.Sequential([ConvTransformer([conv2, conv2], pooling=kr.layers.MaxPooling2D(2, 1, 'same')),
                          ConvTransformer([conv1, conv1], pooling=kr.layers.MaxPooling2D(2, 1, 'same'))])
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
    #     ConvTransformer([kr.layers.Conv2D(64, 3, 2, 'same', activation='relu') for _ in range(2)]),
    #     ConvTransformer([kr.layers.Conv2DTranspose(12, 3, 2, 'same', activation='relu') for _ in range(2)]),
    #     ConvTransformer([kr.layers.Conv2D(48, 3, 2, 'same', activation='relu') for _ in range(2)]),
    #     kr.layers.Flatten(),
    #     kr.layers.Dense(10)
    # ])
    # model.compile(optimizer=kr.optimizers.Adam(),
    #               loss=kr.losses.SparseCategoricalCrossentropy(True),
    #               metrics=[kr.metrics.sparse_categorical_accuracy])
    # model.fit(x_train, y_train, 64, 5, validation_data=(x_test, y_test))

    model.summary()
