import numpy as np
import tensorflow.keras as kr
import tensorflow as tf
from horse2zebra_datasets import Datasets
import cv2


def block(feature, kernel_size=3, strides=2, padding='same', act=tf.nn.leaky_relu):
    return kr.Sequential([
        kr.layers.Conv2D(feature, kernel_size, strides, padding, use_bias=False),
        kr.layers.BatchNormalization(),
        kr.layers.Activation(act)
    ])


def block_t(feature, kernel_size=3, strides=2, padding='same'):
    return kr.Sequential([
        kr.layers.Conv2DTranspose(feature, kernel_size, strides, padding, use_bias=False),
        kr.layers.BatchNormalization(),
        kr.layers.LeakyReLU()
    ])


class UNet(kr.Model):
    def __init__(self):
        super(UNet, self).__init__()
        self.conv1 = block(64)
        self.conv2 = block(128)
        self.conv3 = block(256)
        self.conv4 = block(512)

        self.conv1_t = block(384, strides=1)
        self.conv2_t = block(192, strides=1)
        self.conv3_t = block(96, strides=1)
        self.conv4_t = block(3, strides=1, act=tf.nn.tanh)

        self.t_conv1 = block_t(512)
        self.t_conv2 = block_t(256)
        self.t_conv3 = block_t(128)
        self.t_conv4 = block_t(64)

    def call(self, inputs, training=None, mask=None):
        out1 = self.conv1(inputs)
        out2 = self.conv2(out1)
        out3 = self.conv3(out2)
        out4 = self.conv4(out3)

        out5 = self.t_conv1(out4)
        out6 = tf.concat([out5, out3], axis=-1)
        out7 = self.conv1_t(out6)

        out8 = self.t_conv2(out7)
        out9 = tf.concat([out8, out2], axis=-1)
        out10 = self.conv2_t(out9)
        out10 = kr.layers.Dropout(.1)(out10)

        out11 = self.t_conv3(out10)
        out12 = tf.concat([out11, out1], axis=-1)
        out13 = self.conv3_t(out12)
        out13 = kr.layers.Dropout(.1)(out13)

        out14 = self.t_conv4(out13)
        out15 = tf.concat([out14, inputs], axis=-1)
        out15 = kr.layers.Dropout(.1)(out15)
        out16 = self.conv4_t(out15)
        return out16


class PatchGAN(kr.Model):
    def __init__(self):
        super(PatchGAN, self).__init__()
        # 256*256*3
        self.uconv1 = block(128)
        # 128*128*128
        self.uconv2 = block(256)
        # 64*64*256
        self.uconv3 = block(512)
        # 32*32*512
        self.uconv4 = block(1024)
        # 16*16*1024
        self.dconv1 = block_t(512, strides=1)
        # 16*16*512

        # u3: 32*32*512
        self.uconv5 = block(256)
        # 16*16*256

        # 16*16*(512+256)
        self.dconv2 = block_t(384, strides=1)
        # 16*16*384

        # u2: 64*64*256
        self.uconv6 = block(128, strides=4)
        # 16*16*128

        # 16*16*(384+128)
        self.dconv3 = block_t(256, strides=1)
        # 16*16*256

        # u1: 128*128*128
        self.uconv7 = block(64, strides=8)
        # 16*16*64

        # 16*16(256+64)
        self.dconv4 = block_t(128, strides=1)
        # 16*16*128

        # 256*256*3
        self.uconv8 = block(32, strides=16)
        # 16*16*32

        # 16*16*(128+32)
        self.dconv5 = block_t(64, strides=1)
        # 16*16*64
        self.uconv9 = block(1, strides=1, act=kr.activations.softmax)
        # 16*16*1

    def call(self, inputs, training=None, mask=None):
        # 256*256*3
        out1 = self.uconv1(inputs)
        # 128*128*128
        out2 = self.uconv2(out1)
        # 64*64*256
        out3 = self.uconv3(out2)
        # 32*32*512
        out4 = self.uconv4(out3)
        # 16*16*1024
        out5 = self.dconv1(out4)
        # 16*16*512

        # u3: 32*32*512
        out6 = self.uconv5(out3)
        # 16*16*256

        # 16*16*(512+256)
        out7 = tf.concat([out5, out6], axis=-1)
        out8 = self.dconv2(out7)
        # 16*16*384

        # u2: 64*64*256
        out9 = self.uconv6(out2)
        # 16*16*128

        # 16*16*(384+128)
        out10 = tf.concat([out8, out9], axis=-1)
        out11 = self.dconv3(out10)
        out11 = kr.layers.Dropout(.1)(out11)
        # 16*16*256

        # u1: 128*128*128
        out12 = self.uconv7(out1)
        # 16*16*64

        # 16*16(256+64)
        out13 = tf.concat([out11, out12], axis=-1)
        out14 = self.dconv4(out13)
        out14 = kr.layers.Dropout(.1)(out14)
        # 16*16*128

        # 256*256*3
        out15 = self.uconv8(inputs)
        # 16*16*32

        # 16*16*(128+32)
        out16 = tf.concat([out14, out15], axis=-1)
        out17 = self.dconv5(out16)
        out17 = kr.layers.Dropout(.1)(out17)
        # 16*16*64
        out18 = self.uconv9(out17)
        # 16*16*1
        return out18


if __name__ == '__main__':
    dataset = Datasets()
    horse_data, _ = dataset.load_test_dataset()
    unet = UNet()
    test_data = horse_data[10:21]
    data = unet(test_data)
    data = dataset.convert_data(np.array(data))
    cv2.imshow('data', np.hstack((data[10], dataset.convert_data(test_data[10]))))
    cv2.waitKey()
    cv2.destroyAllWindows()
    patchGAN = PatchGAN()
    data = patchGAN(test_data)
    data = dataset.convert_data(np.array(data))
    cv2.imshow('data', data[7])
    cv2.waitKey()
    cv2.destroyAllWindows()

