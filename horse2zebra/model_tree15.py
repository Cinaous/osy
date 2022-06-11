from abc import ABC
import tensorflow.keras as kr
import tensorflow as tf
from horse2zebra_datasets13 import Datasets, convert
# from face2face_datasets2 import Datasets, convert
import matplotlib.pyplot as plt

swish = kr.layers.Activation(kr.activations.swish)


class Tree(kr.Model, ABC):
    def __init__(self):
        super(Tree, self).__init__()
        # 64
        self.conv1 = kr.Sequential([
            kr.layers.Conv2D(32, 11, 2, 'same'),
            kr.layers.LayerNormalization(),
            swish
        ])
        # 56
        self.conv2 = kr.Sequential([
            kr.layers.Conv2D(48, 9),
            kr.layers.LayerNormalization(),
            swish,
            kr.layers.Dropout(.2)
        ])
        # 28
        self.conv3 = kr.Sequential([
            kr.layers.Conv2D(64, 7, 2, 'same'),
            kr.layers.LayerNormalization(),
            swish
        ])
        # 14
        self.conv4 = kr.Sequential([
            kr.layers.Conv2D(76, 5, 2, 'same'),
            kr.layers.LayerNormalization(),
            swish
        ])
        # 14
        self.conv5 = kr.Sequential([
            kr.layers.Conv2D(32, 9, 4, 'same'),
            kr.layers.LayerNormalization(),
            swish
        ])
        self.last = kr.layers.Conv2D(32, 3, 1, 'same')

    def call(self, inputs, training=None, mask=None):
        x = self.conv1(inputs)
        x_ = x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x_ = self.conv5(x_)
        x = self.last(tf.concat([x, x_], -1))
        return x


class TreeUNet(kr.Model, ABC):
    def __init__(self):
        super(TreeUNet, self).__init__()
        # 64
        self.conv1 = kr.Sequential([
            kr.layers.Conv2D(32, 11, 2, 'same'),
            kr.layers.LayerNormalization(),
            swish
        ])
        # 56
        self.conv2 = kr.Sequential([
            kr.layers.Conv2D(48, 9),
            kr.layers.LayerNormalization(),
            swish,
            kr.layers.Dropout(.2)
        ])
        # 28
        self.conv3 = kr.Sequential([
            kr.layers.Conv2D(64, 7, 2, 'same'),
            kr.layers.LayerNormalization(),
            swish
        ])
        # 14
        self.conv4 = kr.Sequential([
            kr.layers.Conv2D(76, 5, 2, 'same'),
            kr.layers.LayerNormalization(),
            swish
        ])
        # 28
        self.unet1 = kr.Sequential([
            kr.layers.Conv2DTranspose(64, 5, 2, 'same'),
            kr.layers.LayerNormalization(),
            swish
        ])
        # 56
        self.unet2 = kr.Sequential([
            kr.layers.Conv2DTranspose(48, 7, 2, 'same'),
            kr.layers.LayerNormalization(),
            swish
        ])
        # 64
        self.unet3 = kr.Sequential([
            kr.layers.Conv2DTranspose(32, 9),
            kr.layers.LayerNormalization(),
            swish
        ])
        self.unet4 = kr.layers.Conv2DTranspose(3, 11, 2, 'same')

    def call(self, inputs, training=None, mask=None):
        x = self.conv1(inputs)
        d2 = x = self.conv2(x)
        d3 = x = self.conv3(x)
        x = self.conv4(x)
        x = self.unet1(x)
        x = self.unet2(x + d3)
        x = self.unet3(x + d2)
        x = self.unet4(x)
        return x


if __name__ == '__main__':
    dataset = Datasets()
    x_train, x_test = dataset.load_test_dataset()
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
    model.layers[0].summary()
    model.compile(optimizer=kr.optimizers.Adam(),
                  loss=kr.losses.SparseCategoricalCrossentropy(True),
                  metrics=[kr.metrics.sparse_categorical_accuracy])
    horse_label = tf.zeros(1)
    zebra_label = tf.ones(1)
    for horse, zebra in dataset:
        model.fit(tf.concat([horse, zebra], 0), tf.concat([horse_label, zebra_label], 0),
                  validation_data=(x_test, label_test))
