from abc import ABC
import tensorflow.keras as kr
import tensorflow as tf
# from horse2zebra_datasets7 import Datasets
# from horse2zebra_datasets13 import Datasets
# from face2face_datasets1 import Datasets
from face2face_datasets2 import Datasets
import matplotlib.pyplot as plt


def conv_init(units, num=3):
    down = kr.Sequential([
        *[kr.Sequential([
            kr.layers.Conv2D(units * 2 ** ((i + 1) % 2), 3, padding='same'),
            kr.layers.LayerNormalization(),
            kr.layers.ReLU(),
        ]) for i in range(num)],
        kr.layers.Conv2D(4 * units, 3, 2, 'same'),
        kr.layers.LayerNormalization(),
        kr.layers.ReLU(),
        kr.layers.Dropout(.1)
    ])
    return down


def unet_init(units, num=3):
    down = kr.Sequential([
        kr.layers.Conv2DTranspose(units, 3, 2, 'same'),
        kr.layers.LayerNormalization(),
        kr.layers.ReLU(),
        *[kr.Sequential([
            kr.layers.Conv2D(units * 2 ** ((i + 1) % 2), 3, padding='same'),
            kr.layers.LayerNormalization(),
            kr.layers.ReLU(),
            kr.layers.Dropout(.1)
        ]) for i in range(num)]
    ])
    return down


class Tree(kr.Model, ABC):
    def __init__(self):
        super(Tree, self).__init__()
        # 64 192
        self.conv1 = conv_init(48)
        # 32 384
        self.conv2 = conv_init(96)
        # 16 768
        self.conv3 = conv_init(192)
        # 32 192
        self.up = kr.Sequential([
            kr.layers.UpSampling2D(),
            kr.layers.Conv2D(192, 3, padding='same')
        ])
        self.last = kr.Sequential([
            kr.layers.Conv2D(384, 3, padding='same'),
            kr.layers.LayerNormalization(),
            kr.layers.ReLU(),
            kr.layers.Conv2D(2, 3, padding='same')
        ])

    def call(self, inputs, training=None, mask=None):
        x = self.conv1(inputs)
        _x = self.conv2(x)
        x = self.conv3(_x)
        x_ = self.up(x)
        x = tf.concat([_x, x_], -1)
        x = self.last(x)
        return x


class TreeUNet(kr.Model, ABC):
    def __init__(self):
        super(TreeUNet, self).__init__()
        # 64 192
        self.conv1 = conv_init(48)
        # 32 384
        self.conv2 = conv_init(96)
        # 16 768
        self.conv3 = conv_init(192)
        # 8 1536
        self.conv4 = conv_init(384)
        # 4 1536
        self.conv5 = conv_init(384)
        # 8 768
        self.unet1 = unet_init(384)
        # 16 384
        self.unet2 = unet_init(192)
        # 32 192
        self.unet3 = unet_init(96)
        # 64 96
        self.unet4 = unet_init(48)
        # 128 3
        self.unet5 = kr.Sequential([
            kr.layers.UpSampling2D(),
            kr.layers.Conv2D(48, 3, padding='same'),
            kr.layers.LayerNormalization(),
            kr.layers.ReLU(),
            kr.layers.Conv2D(3, 3, padding='same')
        ])

    def call(self, inputs, training=None, mask=None):
        x1 = self.conv1(inputs)
        x2 = self.conv2(x1)
        x3 = self.conv3(x2)
        x4 = self.conv4(x3)
        x5 = self.conv5(x4)
        u1 = self.unet1(x5)
        u1 = tf.concat([u1, x4], -1)
        u2 = self.unet2(u1)
        u2 = tf.concat([u2, x3], -1)
        u3 = self.unet3(u2)
        u3 = tf.concat([u3, x2], -1)
        u4 = self.unet4(u3)
        u4 = tf.concat([u4, x1], -1)
        u5 = self.unet5(u4)
        return u5


if __name__ == '__main__':
    dataset = Datasets(1)
    x_train, x_test = dataset.load_test_dataset()
    x_train, x_test = dataset.scaler.fit_transform(x_train), dataset.scaler.fit_transform(x_test)
    x_test = tf.concat([x_train, x_test], 0)
    label_test = tf.concat([tf.zeros(10), tf.ones(10)], 0)
    unet = TreeUNet()
    xp = unet(x_train)
    unet.summary()
    print(xp.shape)
    plt.subplot(121)
    plt.imshow(dataset.convert_data(x_train[-1]))
    plt.subplot(122)
    plt.imshow(dataset.convert_data(xp[0].numpy()))
    plt.show()
    model = kr.Sequential([
        Tree(),
        kr.layers.Flatten(),
        kr.layers.Dense(2)
    ])
    model.build([None, 128, 128, 3])
    model.summary()
    model.compile(optimizer=kr.optimizers.Adam(),
                  loss=kr.losses.SparseCategoricalCrossentropy(True),
                  metrics=[kr.metrics.sparse_categorical_accuracy])
    horse_label = tf.zeros(1)
    zebra_label = tf.ones(1)
    for horse, zebra in dataset:
        model.fit(tf.concat([horse, zebra], 0), tf.concat([horse_label, zebra_label], 0),
                  validation_data=(x_test, label_test))
