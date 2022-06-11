import os.path as path
from abc import ABC
import cv2
import numpy as np
import tensorflow.keras as kr
from face2face_datasets2 import Datasets, convert
from models.conv_rnn import ConvRNN
import copy

dataset = Datasets()
DH = kr.Sequential([
    ConvRNN(64, 11),
    ConvRNN(96, 9),
    ConvRNN(128, 7),
    ConvRNN(160, 5),
    ConvRNN(192, 5, 1, activation=None)
])
DZ = copy.deepcopy(DH)
H2Z = kr.Sequential([
    copy.deepcopy(DH),
    ConvRNN(160, 5, 1, conv=kr.layers.Conv2DTranspose),
    ConvRNN(128, 5, conv=kr.layers.Conv2DTranspose),
    ConvRNN(96, 7, conv=kr.layers.Conv2DTranspose),
    ConvRNN(64, 9, conv=kr.layers.Conv2DTranspose),
    ConvRNN(3, 11, conv=kr.layers.Conv2DTranspose, activation=None)
])
Z2H = copy.deepcopy(H2Z)
epoch, first = 0, True


class GModel(kr.Model, ABC):
    def __init__(self):
        super(GModel, self).__init__()
        self.h2z = H2Z
        self.z2h = Z2H
        self.compile(optimizer=kr.optimizers.Adam(2e-4),
                     loss=kr.losses.mae,
                     loss_weights=[10, 10, 1, 1])

    def call(self, inputs, training=None, mask=None):
        horse, zebra = inputs
        h2z = self.h2z(horse)
        h2z2h = self.z2h(h2z)
        z2h = self.z2h(zebra)
        z2h2z = self.h2z(z2h)
        return h2z2h, z2h2z, DH(z2h), DZ(h2z)


class DModel(kr.Model, ABC):
    def __init__(self):
        super(DModel, self).__init__()
        self.dh = DH
        self.dz = DZ
        self.compile(optimizer=kr.optimizers.Adam(2e-4),
                     loss=kr.losses.mae)

    def call(self, inputs, training=None, mask=None):
        horse, zebra, z2h, h2z = inputs
        return self.dh(horse), self.dz(zebra), self.dh(z2h), self.dz(h2z)


gmodel = GModel()
dmodel = DModel()

real_label = np.ones([dataset.batch, 8, 8, 192])
fake_label = np.zeros([dataset.batch, 8, 8, 192])
dir_label = '77'
G_save_path = 'outputs/G%s.ckpt' % dir_label
D_save_path = 'outputs/D%s.ckpt' % dir_label
# if path.exists(G_save_path + '.index'):
#     gmodel.load_weights(G_save_path)
#     dmodel.load_weights(D_save_path)
for horse, zebra in dataset:
    if epoch % 25 == 0 or first:
        test_horse, test_zebra = dataset.load_test_dataset(1)
        h2z = H2Z(test_horse)
        h2z2h = Z2H(h2z)
        img1 = np.concatenate((test_horse, h2z, h2z2h), axis=2)[0]
        z2h = Z2H(test_zebra)
        z2h2z = H2Z(z2h)
        img2 = np.concatenate((test_zebra, z2h, z2h2z), axis=2)[0]
        img3 = np.vstack((img1, img2))
        img3 = convert(img3)
        if first:
            first = False
            cv2.imshow('trainA&trainB', img3)
            cv2.waitKey()
            cv2.destroyAllWindows()
        cv2.imwrite('outputs/images%s/trainA&z_%d.jpg' % (dir_label, epoch), img3)
    gmodel.fit([horse, zebra], [horse, zebra, real_label, real_label], workers=12, use_multiprocessing=True)
    dmodel.fit([horse, zebra, Z2H(zebra), H2Z(horse)],
               [real_label, real_label, fake_label, fake_label], workers=12, use_multiprocessing=True)
    try:
        gmodel.save_weights(G_save_path)
        dmodel.save_weights(D_save_path)
    except Exception as ex:
        print(ex.args)
    epoch += 1
    print('current epoch is %d\r\n' % epoch)
