from abc import ABC
import tensorflow.keras as kr
import tensorflow as tf
from horse2zebra_datasets8 import Datasets
# from horse2zebra_datasets13 import Datasets
from face2face_datasets1 import Datasets
# from face2face_datasets2 import Datasets
import matplotlib.pyplot as plt


class Tree(kr.Model, ABC):
    def __init__(self):
        super(Tree, self).__init__()
        # 64
        self.conv1 = kr.Sequential([
            kr.layers.Conv2D(64, 11, 2, 'same'),
            kr.layers.LayerNormalization(),
            kr.layers.PReLU()
        ])
        # 56
        self.conv2 = kr.Sequential([
            kr.layers.Conv2D(64, 9),
            kr.layers.LayerNormalization(),
            kr.layers.PReLU(),
            kr.layers.Dropout(.2)
        ])
        # 28
        self.conv3 = kr.Sequential([
            kr.layers.Conv2D(128, 7, 2, 'same'),
            kr.layers.LayerNormalization(),
            kr.layers.PReLU()
        ])
        # 14
        self.conv4 = kr.Sequential([
            kr.layers.Conv2D(128, 5, 2, 'same'),
            kr.layers.LayerNormalization(),
            kr.layers.PReLU()
        ])
        # 28 -14 = 14
        self.conv5 = kr.Sequential([
            kr.layers.Conv2D(128, 15),
            kr.layers.LayerNormalization(),
            kr.layers.PReLU(),
            kr.layers.Dropout(.2)
        ])
        self.last = kr.layers.Conv2D(32, 5, 1, 'same')

    def call(self, inputs, training=None, mask=None):
        x = self.conv1(inputs)
        x = self.conv2(x)
        x3 = self.conv3(x)
        x = self.conv4(x3)
        x5 = self.conv5(x3)
        x = tf.concat([x, x5], -1)
        x = self.last(x)
        return x


class TreeUNet(kr.Model, ABC):
    def __init__(self):
        super(TreeUNet, self).__init__()
        # 64
        self.conv1 = kr.Sequential([
            kr.layers.Conv2D(64, 11, 2, 'same'),
            kr.layers.LayerNormalization(),
            kr.layers.PReLU()
        ])
        # 56
        self.conv2 = kr.Sequential([
            kr.layers.Conv2D(64, 9),
            kr.layers.LayerNormalization(),
            kr.layers.PReLU(),
            kr.layers.Dropout(.2)
        ])
        # 28
        self.conv3 = kr.Sequential([
            kr.layers.Conv2D(128, 7, 2, 'same'),
            kr.layers.LayerNormalization(),
            kr.layers.PReLU()
        ])
        # 14
        self.conv4 = kr.Sequential([
            kr.layers.Conv2D(128, 5, 2, 'same'),
            kr.layers.LayerNormalization(),
            kr.layers.PReLU()
        ])
        # 28
        self.unet1 = kr.Sequential([
            kr.layers.Conv2DTranspose(128, 5, 2, 'same'),
            kr.layers.LayerNormalization(),
            kr.layers.PReLU()
        ])
        # 56
        self.unet2 = kr.Sequential([
            kr.layers.Conv2DTranspose(64, 7, 2, 'same'),
            kr.layers.LayerNormalization(),
            kr.layers.PReLU()
        ])
        # 64
        self.unet3 = kr.Sequential([
            kr.layers.Conv2DTranspose(64, 9),
            kr.layers.LayerNormalization(),
            kr.layers.PReLU()
        ])
        self.unet4 = kr.layers.Conv2DTranspose(3, 11, 2, 'same')

    def call(self, inputs, training=None, mask=None):
        d1 = self.conv1(inputs)
        d2 = self.conv2(d1)
        d3 = self.conv3(d2)
        d4 = self.conv4(d3)
        u1 = self.unet1(d4)
        u2 = self.unet2(u1 + d3)
        u3 = self.unet3(u2 + d2)
        u4 = self.unet4(u3)
        return u4


if __name__ == '__main__':
    dataset = Datasets(1)
    x_train, x_test = dataset.load_test_dataset()
    # x_train, x_test = dataset.scaler.fit_transform(x_train), dataset.scaler.fit_transform(x_test)
    x_test = tf.concat([x_train, x_test], 0)
    label_test = tf.concat([tf.zeros(10), tf.ones(10)], 0)
    unet = TreeUNet()
    xp = unet(x_train)
    unet.summary()
    print(xp.shape)
    plt.subplot(121)
    plt.imshow(dataset.convert_data(x_train[-1]))
    plt.subplot(122)
    plt.imshow(dataset.convert_data(xp[-1].numpy()))
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
