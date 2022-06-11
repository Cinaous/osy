from abc import ABC
import tensorflow.keras as kr
import tensorflow as tf
import numpy as np


class Discriminator(kr.Model, ABC):
    def __init__(self, features, classifer=2, last_ksize=3, first_ksize=7):
        super(Discriminator, self).__init__()
        self.downs = []
        for i, feature in enumerate(features):
            if i == 0:
                self.downs.append(Sampling(feature, kernel_size=first_ksize, normalize=False))
            elif i == len(features) - 1:
                self.downs.append(Sampling(feature, strides=1))
            else:
                self.downs.append(Sampling(feature))
        self.last = kr.layers.Conv2D(classifer, last_ksize, activation=kr.activations.softmax)

    def call(self, inputs, training=None, mask=None):
        for down in self.downs:
            inputs = down(inputs)
        return self.last(inputs)


class UNet(kr.Model, ABC):
    def __init__(self, features, channel=3, last_ksize=3, first_ksize=7):
        super(UNet, self).__init__()
        self.downs = []
        for i, feature in enumerate(features):
            if i == 0:
                self.downs.append(Sampling(feature, kernel_size=first_ksize, normalize=False))
            else:
                self.downs.append(Sampling(feature))
        self.ups = []
        for i, feature in enumerate(features[:-1][::-1]):
            if i == 0:
                self.ups.append(Sampling(feature, False, normalize=False, drop_out=.5))
            else:
                self.ups.append(Sampling(feature, False))
        self.last = kr.layers.Conv2DTranspose(channel, last_ksize, 2, 'same', activation=kr.activations.tanh)

    def call(self, inputs, training=None, mask=None):
        skips = []
        for down in self.downs:
            inputs = down(inputs)
            skips.append(inputs)
        skips = reversed(skips[:-1])
        for skip, up in zip(skips, self.ups):
            inputs = tf.concat([skip, up(inputs)], axis=-1)
        return self.last(inputs)


class Sampling(kr.Model, ABC):
    def __init__(self, filters, down=True, kernel_size=5, strides=2, padding='same', normalize=True,
                 activation=kr.layers.LeakyReLU, drop_out=None, pooling=None, **kwargs):
        super(Sampling, self).__init__()
        self.conv = kr.Sequential()
        if down:
            self.conv.add(kr.layers.Conv2D(filters, kernel_size, strides, padding, **kwargs,
                                           kernel_initializer=kr.initializers.random_uniform(minval=0, maxval=.02)))
        else:
            self.conv.add(kr.layers.Conv2DTranspose(filters, kernel_size, strides, padding, **kwargs,
                                                    kernel_initializer=kr.initializers.random_uniform(minval=0,
                                                                                                      maxval=.02)))
        if normalize:
            self.conv.add(kr.layers.LayerNormalization())
        if activation is not None:
            self.conv.add(activation())
        if pooling is not None:
            self.conv.add(pooling())
        if drop_out is not None:
            self.conv.add(kr.layers.Dropout(drop_out))

    def call(self, inputs, training=None, mask=None):
        out = self.conv(inputs)
        return out


class GCModel(kr.Model, ABC):
    def __init__(self, G: kr.Model, F: kr.Model, Dx: kr.Model, Dy: kr.Model):
        super(GCModel, self).__init__()
        self.G = G
        self.F = F
        self.Dx = Dx
        self.Dy = Dy
        self.Dx.trainable = False
        self.Dy.trainable = False
        if not self.G.trainable:
            self.G.trainable = True
        if not self.F.trainable:
            self.F.trainable = True

    def call(self, inputs, training=None, mask=None):
        gy = self.G(inputs[0])
        dy = self.Dy(gy)
        gx = self.F(inputs[1])
        dx = self.Dx(gx)
        x2y2x = self.F(gy)
        y2x2y = self.G(gx)
        return dy, dx, x2y2x, y2x2y


class DCModel(kr.Model, ABC):
    def __init__(self, G: kr.Model, F: kr.Model, Dx: kr.Model, Dy: kr.Model):
        super(DCModel, self).__init__()
        self.G = G
        self.F = F
        self.Dx = Dx
        self.Dy = Dy
        self.G.trainable = False
        self.F.trainable = False
        if not self.Dx.trainable:
            self.Dx.trainable = True
        if not self.Dy.trainable:
            self.Dy.trainable = True

    def call(self, inputs, training=None, mask=None):
        gy = self.G(inputs[0])
        dy = self.Dy(gy)
        gx = self.F(inputs[1])
        dx = self.Dx(gx)
        return dy, dx, self.Dy(inputs[1]), self.Dx(inputs[0])


if __name__ == '__main__':
    inputs = np.random.uniform(size=[5, 128, 128, 3])
    unet = UNet([64, 128, 256, 512, 512, 512, 512])
    outputs = unet(inputs)
    print(outputs.shape)

    discriminator = Discriminator([128, 256, 512])
    outputs = discriminator(inputs)
    print(outputs.shape)
