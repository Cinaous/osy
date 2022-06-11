import os.path as path
from abc import ABC
import cv2
import numpy as np
import tensorflow.keras as kr
from face2face_datasets2 import Datasets, convert
from beauty.beauty_model9 import SplitDimsLayer
from cache_utils import Cache
import copy


class Encoder(kr.Model, ABC):
    def __init__(self):
        super(Encoder, self).__init__()
        self.conv = [SplitDimsLayer(1 + i, 48, 5, 2) for i in range(4)]
        self.conv.append(SplitDimsLayer(5, 48, 5, 2, activation=None))

    def call(self, inputs, training=None, mask=None):
        x = inputs
        for conv in self.conv:
            x = conv(x)
        return x


class Decoder(kr.Model, ABC):
    def __init__(self):
        super(Decoder, self).__init__()
        self.down = [SplitDimsLayer(1 + i, 48, 5, 2) for i in range(5)]
        self.up = [SplitDimsLayer(4 - i, 48, 5, 2, conv=kr.layers.Conv2DTranspose) for i in range(4)]
        self.last = kr.layers.Conv2DTranspose(3, 5, 2, 'same')

    def call(self, inputs, training=None, mask=None):
        x = inputs
        downs = []
        for down in self.down:
            x = down(x)
            downs.append(x)
        downs = downs[:-1][::-1]
        for up, down in zip(self.up, downs):
            x = up(x)
            x += down
        x = self.last(x)
        return x


dataset = Datasets()
DH = Encoder()
DZ = copy.deepcopy(DH)
H2Z = Decoder()
Z2H = copy.deepcopy(H2Z)
epoch, first, h_cache, z_cache = 0, True, Cache(), Cache()


class GModel(kr.Model, ABC):
    def __init__(self):
        super(GModel, self).__init__()
        self.h2z = H2Z
        self.z2h = Z2H
        self.compile(optimizer=kr.optimizers.Adam(2e-4),
                     loss=kr.losses.mae,
                     loss_weights=[10, 10, 1, 1, 5, 5])

    def call(self, inputs, training=None, mask=None):
        horse, zebra = inputs
        h2z = self.h2z(horse)
        h2z2h = self.z2h(h2z)
        z2h = self.z2h(zebra)
        z2h2z = self.h2z(z2h)
        h2h = self.z2h(horse)
        z2z = self.h2z(zebra)
        return h2z2h, z2h2z, DH(z2h), DZ(h2z), h2h, z2z


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

real_label = None
fake_label = None
dir_label = '78'
G_save_path = 'outputs/G%s.ckpt' % dir_label
D_save_path = 'outputs/D%s.ckpt' % dir_label
if path.exists(G_save_path + '.index'):
    gmodel.load_weights(G_save_path)
    dmodel.load_weights(D_save_path)
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
            Z2H.summary()
            dh = DH(horse)
            DH.summary()
            d_shape = dh.shape[1:]
            real_label = real_label or np.ones([dataset.batch, *d_shape])
            fake_label = fake_label or np.zeros([dataset.batch, *d_shape])
            first = False
            cv2.imshow('trainA&trainB', img3)
            cv2.waitKey()
            cv2.destroyAllWindows()
        cv2.imwrite('outputs/images%s/trainA&z_%d.jpg' % (dir_label, epoch), img3)
    gmodel.fit([horse, zebra], [horse, zebra, real_label, real_label, horse, zebra],
               workers=12, use_multiprocessing=True)
    dmodel.fit([horse, zebra, h_cache(Z2H(zebra)), z_cache(H2Z(horse))],
               [real_label, real_label, fake_label, fake_label],
               workers=12, use_multiprocessing=True)
    try:
        gmodel.save_weights(G_save_path)
        dmodel.save_weights(D_save_path)
    except Exception as ex:
        print(ex.args)
    epoch += 1
    print('current epoch is %d\r\n' % epoch)
