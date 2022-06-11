from abc import ABC
import tensorflow.keras as kr


class Encoder(kr.Model, ABC):
    def __init__(self):
        super(Encoder, self).__init__()
        # 32 * 128
        self.conv1 = kr.Sequential([
            kr.layers.Conv2D(128, 3, 2, 'same', activation=kr.activations.relu),
            kr.layers.MaxPool2D(strides=2, padding='same'),
        ])
        # 8 * 256
        self.conv2 = kr.Sequential([
            kr.layers.Conv2D(256, 3, 2, 'same'),
            kr.layers.LayerNormalization(epsilon=1e-5),
            kr.layers.ReLU(),
            kr.layers.MaxPool2D(strides=2, padding='same'),
            kr.layers.Dropout(.1)
        ])
        # 4 * 512
        self.conv3 = kr.Sequential([
            kr.layers.Conv2D(512, 3, 2, 'same'),
            kr.layers.LayerNormalization(epsilon=1e-5),
            kr.layers.ReLU(),
            kr.layers.Dropout(.2)
        ])
        # 1 * 1024
        self.conv4 = kr.Sequential([
            kr.layers.Conv2D(1024, 3, 2, 'same'),
            kr.layers.MaxPool2D(strides=2, padding='same')
        ])

    def call(self, inputs, training=None, mask=None):
        x = self.conv1(inputs)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        return x


class Decoder(kr.Model, ABC):
    def __init__(self):
        super(Decoder, self).__init__()
        # 4 * 512
        self.conv1 = kr.Sequential([
            kr.layers.Conv2DTranspose(1024, 3, 2, 'same', activation=kr.activations.relu),
            kr.layers.Conv2DTranspose(512, 3, 2, 'same', activation=kr.activations.relu),
        ])
        # 16 * 128
        self.conv2 = kr.Sequential([
            kr.layers.Conv2DTranspose(256, 3, 2, 'same'),
            kr.layers.LayerNormalization(epsilon=1e-5),
            kr.layers.ReLU(),
            kr.layers.Conv2DTranspose(128, 3, 2, 'same', activation=kr.activations.relu),
            kr.layers.Dropout(.1)
        ])
        # 32 * 64
        self.conv3 = kr.Sequential([
            kr.layers.Conv2DTranspose(64, 3, 2, 'same'),
            kr.layers.LayerNormalization(epsilon=1e-5),
            kr.layers.ReLU(),
            kr.layers.Dropout(.2)
        ])
        # 128 * 3
        self.conv4 = kr.Sequential([
            kr.layers.Conv2DTranspose(32, 3, 2, 'same'),
            kr.layers.LayerNormalization(epsilon=1e-5),
            kr.layers.ReLU(),
            kr.layers.Conv2DTranspose(3, 3, 2, 'same')
        ])

    def call(self, inputs, training=None, mask=None):
        x = self.conv1(inputs)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        return x
