from abc import ABC
import tensorflow.keras as kr
import tensorflow as tf
# from horse2zebra_datasets8 import Datasets
# from horse2zebra_datasets13 import Datasets
# from face2face_datasets1 import Datasets
from face2face_datasets2 import Datasets, convert
import matplotlib.pyplot as plt

swish = kr.layers.Activation(kr.activations.swish)


def conv_init(units, num=1, activation=True):
    layers = [kr.layers.Conv2D(units, 3, 2, 'same'),
              *[kr.Sequential([
                  kr.layers.LayerNormalization(),
                  kr.layers.Dense(2 * units, activation='swish'),
                  kr.layers.Conv2D(units, 3, 1, 'same')
              ]) for _ in range(num)]]
    layers = [*layers,
              kr.layers.LayerNormalization(),
              swish] if activation else layers
    return kr.Sequential(layers)


def unet_init(units, num=1, activation=True):
    layers = [kr.layers.Conv2DTranspose(units, 3, 2, 'same'),
              *[kr.Sequential([
                  kr.layers.LayerNormalization(),
                  kr.layers.Dense(2 * units, activation='swish'),
                  kr.layers.Conv2DTranspose(units, 3, 1, 'same')
              ]) for _ in range(num)]]
    layers = [*layers,
              kr.layers.LayerNormalization(),
              swish] if activation else layers
    return kr.Sequential(layers)


class Tree(kr.Model, ABC):
    def __init__(self):
        super(Tree, self).__init__()
        self.model = kr.Sequential([
            *[conv_init(64, 1 + 2 * i, i != 3) for i in range(4)],
        ])

    def call(self, inputs, training=None, mask=None):
        x = self.model(inputs)
        return x


class TreeUNet(kr.Model, ABC):
    def __init__(self):
        super(TreeUNet, self).__init__()
        # 64
        self.conv1 = conv_init(64)
        # 32
        self.conv2 = conv_init(64, 3)
        # 16
        self.conv3 = conv_init(64, 5)
        # 8
        self.conv4 = conv_init(64, 7)
        # 16
        self.unet1 = unet_init(64, 7)
        # 32
        self.unet2 = unet_init(64, 5)
        # 64
        self.unet3 = unet_init(64, 3)
        # 128
        self.unet4 = unet_init(3, activation=False)

    def call(self, inputs, training=None, mask=None):
        d1 = x = self.conv1(inputs)
        d2 = x = self.conv2(x)
        d3 = x = self.conv3(x)
        x = self.conv4(x)
        x = self.unet1(x)
        x = self.unet2(x + d3)
        x = self.unet3(x + d2)
        x = self.unet4(x + d1)
        return x


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
    plt.imshow(convert(x_train[-1]))
    plt.subplot(122)
    plt.imshow(convert(xp[-1].numpy()))
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
