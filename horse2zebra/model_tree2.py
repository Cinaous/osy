from abc import ABC
import tensorflow.keras as kr
import tensorflow as tf
from horse2zebra_datasets6 import Datasets
# from horse2zebra_datasets12 import Datasets
# from face2face_datasets2 import Datasets
import matplotlib.pyplot as plt


def conv_init(units, num):
    down = kr.Sequential([
        *[kr.Sequential([
            kr.layers.Conv2D(units, 3),
            kr.layers.LayerNormalization(epsilon=1e-5),
            kr.layers.Activation(kr.activations.tanh)
        ]) for _ in range(num)],
        kr.layers.MaxPool2D(),
        kr.layers.Dropout(.2)
    ])
    return down


def unet_init(units, num):
    down = kr.Sequential([
        kr.layers.UpSampling2D(),
        *[kr.Sequential([
            kr.layers.Conv2DTranspose(units, 3),
            kr.layers.LayerNormalization(epsilon=1e-5),
            kr.layers.Activation(kr.activations.tanh)
        ]) for _ in range(num)],
        kr.layers.Dropout(.2)
    ])
    return down


class Tree(kr.Model, ABC):
    def __init__(self):
        super(Tree, self).__init__()
        # 62
        self.conv1 = conv_init(48, 2)
        # 28
        self.conv2 = [conv_init(96, 3) for _ in range(2)]
        # 12
        self.conv3 = [conv_init(192, 2) for _ in range(4)]
        # 5
        self.conv4 = [conv_init(384, 1) for _ in range(8)]
        self.dense = self.add_weight(name='csb', shape=[8, 384, 2], dtype=self.dtype,
                                     initializer='glorot_uniform', trainable=True)

    def call(self, inputs, training=None, mask=None):
        x = self.conv1(inputs)
        x = [conv(x) for conv in self.conv2]
        x = [conv(x[i // 2]) for i, conv in enumerate(self.conv3)]
        x = [conv(x[i // 2]) for i, conv in enumerate(self.conv4)]
        x = tf.stack(x, -1)
        x = tf.einsum('...hwcs,sco->...hwo', x, self.dense)
        return x


class TreeUNet(kr.Model, ABC):
    def __init__(self):
        super(TreeUNet, self).__init__()
        # 62
        self.conv1 = conv_init(48, 2)
        # 28
        self.conv2 = [conv_init(96, 3) for _ in range(2)]
        # 12
        self.conv3 = [conv_init(192, 2) for _ in range(4)]
        # 5
        self.conv4 = [conv_init(384, 1) for _ in range(8)]
        # 12
        self.unet1 = [unet_init(192, 1) for _ in range(4)]
        # 28
        self.unet2 = [unet_init(96, 2) for _ in range(2)]
        # 62
        self.unet3 = unet_init(48, 3)
        # 128
        self.unet4 = kr.Sequential([
            kr.layers.UpSampling2D(),
            kr.layers.Conv2DTranspose(3, 3),
            kr.layers.Conv2DTranspose(3, 3)
        ])

    def call(self, inputs, training=None, mask=None):
        x1 = self.conv1(inputs)
        x2 = [conv(x1) for conv in self.conv2]
        x3 = [conv(x2[i // 2]) for i, conv in enumerate(self.conv3)]
        x4 = [conv(x3[i // 2]) for i, conv in enumerate(self.conv4)]
        u1 = [unet(tf.concat(x4[i * 2:2 * (i + 1)], -1)) for i, unet in enumerate(self.unet1)]
        x = [u + x for u, x in zip(u1, x3)]
        u2 = [unet(tf.concat(x[i * 2:2 * (i + 1)], -1)) for i, unet in enumerate(self.unet2)]
        x = [u + x for u, x in zip(u2, x2)]
        u3 = self.unet3(tf.concat(x, -1))
        x = u3 + x1
        x = self.unet4(x)
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
