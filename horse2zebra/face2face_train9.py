import os.path as path
import cv2
import numpy as np
import os
import tensorflow as tf
import tensorflow.keras as kr
from horse2zebra_datasets9 import Datasets
from models_model.model_models11 import UNet, Discriminator, DCModel, GCModel

threads = os.cpu_count() // 4
tf.config.threading.set_intra_op_parallelism_threads(threads)
tf.config.threading.set_inter_op_parallelism_threads(threads)

batch_size = 100
dataset = Datasets(batch=batch_size)
Z2H = UNet([64, 128, 256, 512, 512, 512, 512])
H2Z = UNet([64, 128, 256, 512, 512, 512, 512])
DH = Discriminator([64, 128, 256, 512])
DZ = Discriminator([64, 128, 256, 512])

GC = GCModel(H2Z, Z2H, DH, DZ)
GC.compile(optimizer=kr.optimizers.Adam(),
           loss=[kr.losses.sparse_categorical_crossentropy,
                 kr.losses.sparse_categorical_crossentropy,
                 kr.losses.mae, kr.losses.mae],
           loss_weights=[1., 1., 5., 5.])
DC = DCModel(H2Z, Z2H, DH, DZ)
DC.compile(optimizer=kr.optimizers.Adam(),
           loss=kr.losses.sparse_categorical_crossentropy)

real_label = None
fake_label = None

GC_save_path = 'outputs/GC13.ckpt'
epoch_save_path = 'outputs/images13/epoch.npy'
epoch, first = 0, True
if path.exists(epoch_save_path):
    epoch = np.load(epoch_save_path).max()
    print('load GC weights...')
    GC.load_weights(GC_save_path)

for horse, zebra in dataset:
    if epoch % 2 == 0 or first:
        test_horse, test_zebra = horse[-1:], zebra[-1:]
        h2z = H2Z(test_horse)
        h2z2h = Z2H(h2z)
        img1 = np.concatenate((test_horse, h2z, h2z2h), axis=2)[0]
        z2h = Z2H(test_zebra)
        z2h2z = H2Z(z2h)
        img2 = np.concatenate((test_zebra, z2h, z2h2z), axis=2)[0]
        img3 = np.vstack((img1, img2))
        img3 = Datasets.convert_data(img3)
        if first:
            first = False
            d_shape = DH(test_zebra).shape[1:-1]
            real_label = np.ones([batch_size, *d_shape])
            fake_label = np.zeros([batch_size, *d_shape])
            cv2.imshow('trainA&trainB', img3)
            cv2.waitKey()
            cv2.destroyAllWindows()
        cv2.imwrite('outputs/images13/trainA&z_%d.jpg' % epoch, img3)
    DC.fit([horse, zebra], [fake_label, fake_label, real_label, real_label], 1)
    np.random.shuffle(horse)
    GC.fit([horse, zebra], [real_label, real_label, horse, zebra], 1)
    try:
        GC.save_weights(GC_save_path)
    except Exception:
        pass
    epoch += 1
    np.save(epoch_save_path, epoch)
    print('Current epoch is %d' % epoch)
