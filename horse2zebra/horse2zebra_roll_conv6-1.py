import os.path as path
from abc import ABC
import cv2
import numpy as np
import tensorflow.keras as kr
from cache_utils import Cache
from horse2zebra_datasets13 import Datasets, convert
from model_tree15 import Tree, TreeUNet

dataset = Datasets()
H2Z = TreeUNet()
Z2H = TreeUNet()
DH = Tree()
DZ = Tree()
epoch, first, h_cache, z_cache = 0, True, Cache(), Cache()


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

real_label = np.ones([dataset.batch, 14, 14, 32])
fake_label = np.zeros([dataset.batch, 14, 14, 32])
dir_label = '70'
G_save_path = 'outputs/G%s.ckpt' % dir_label
D_save_path = 'outputs/D%s.ckpt' % dir_label
if path.exists(G_save_path + '.index'):
    gmodel.load_weights(G_save_path)
    dmodel.load_weights(D_save_path)
for horse, zebra in dataset:
    if epoch % 25 == 0 or first:
        test_horse, test_zebra = horse, zebra
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
    gmodel.fit([horse, zebra], [horse, zebra, real_label, real_label],
               workers=6)
    dmodel.fit([h_cache(horse), z_cache(zebra), Z2H(zebra), H2Z(horse)],
               [real_label, real_label, fake_label, fake_label], workers=2)
    try:
        gmodel.save_weights(G_save_path)
        dmodel.save_weights(D_save_path)
    except Exception as ex:
        print(ex.args)
    epoch += 1
    print('current epoch is %d\r\n' % epoch)
