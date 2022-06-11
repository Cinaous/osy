from abc import ABC
import tensorflow.keras as kr
import tensorflow as tf
# from horse2zebra_datasets8 import Datasets
# from horse2zebra_datasets13 import Datasets
# from face2face_datasets1 import Datasets
from face2face_datasets2 import Datasets
import matplotlib.pyplot as plt

activation = kr.layers.Activation(kr.activations.tanh)


def conv_init(units, num):
    return [kr.Sequential([
        kr.layers.Conv2D(units, 3),
        kr.layers.LayerNormalization(),
        activation,
        kr.layers.Dropout(.2)
    ]) for _ in range(num)]


def unet_init(units, num):
    return [kr.Sequential([
        kr.layers.Conv2DTranspose(units, 3),
        kr.layers.LayerNormalization(),
        activation,
        kr.layers.Dropout(.2)
    ]) for _ in range(num)]


class Tree(kr.Model, ABC):
    def __init__(self):
        super(Tree, self).__init__()
        # 50
        self.convs = kr.Sequential([
            # 100
            *conv_init(128, 14),
            # 50
            kr.layers.Conv2D(512, 3, 2, 'same'),
            # 30
            *conv_init(512, 10),
            # 30
            kr.layers.Conv2D(2, 3, padding='same')
        ])

    def call(self, inputs, training=None, mask=None):
        x = self.convs(inputs)
        return x


class TreeUNet(kr.Model, ABC):
    def __init__(self):
        super(TreeUNet, self).__init__()
        self.down1 = kr.Sequential([
            # 100
            *conv_init(128, 14),
            # 50
            kr.layers.Conv2D(512, 3, 2, 'same')
        ])
        self.down2 = kr.Sequential([
            # 30
            *conv_init(512, 10)
        ])
        self.up1 = kr.Sequential([
            # 50
            *unet_init(512, 10)
        ])
        self.up2 = kr.Sequential([
            # 100
            kr.layers.Conv2DTranspose(128, 3, 2, 'same'),
            # 128
            *unet_init(128, 14),
            kr.layers.Conv2DTranspose(3, 3, padding='same')
        ])

    def call(self, inputs, training=None, mask=None):
        down = self.down1(inputs)
        x = self.down2(down)
        up = self.up1(x)
        x = down + up
        x = self.up2(x)
        return x


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
