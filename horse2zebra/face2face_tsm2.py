import os.path as path
import cv2
import numpy as np
import os
import tensorflow as tf
import tensorflow.keras as kr
from face2face_datasets1 import Datasets
from trm.swin_transformer3 import SwinTransformer
from models_model.model_models11 import DCModel, GCModel

threads = os.cpu_count() // 2
tf.config.threading.set_intra_op_parallelism_threads(threads)
tf.config.threading.set_inter_op_parallelism_threads(threads)

batch_size = 100
dataset = Datasets(batch=batch_size)
Z2H = kr.Sequential([
    SwinTransformer(in_channel=6, revert=True),
    SwinTransformer(revert=True)
])
H2Z = kr.Sequential([
    SwinTransformer(in_channel=6, revert=True),
    SwinTransformer(revert=True)
])
DH = kr.Sequential([
    SwinTransformer(in_channel=6),
    SwinTransformer(in_channel=2, revert=True)
])
DZ = kr.Sequential([
    SwinTransformer(in_channel=6),
    SwinTransformer(in_channel=2, revert=True)
])

GC = GCModel(H2Z, Z2H, DH, DZ)
GC.compile(optimizer=kr.optimizers.Adam(2e-5, .5),
           loss=[kr.losses.SparseCategoricalCrossentropy(from_logits=True),
                 kr.losses.SparseCategoricalCrossentropy(from_logits=True),
                 kr.losses.mae, kr.losses.mae],
           loss_weights=[1., 1., 10., 10.])
DC = DCModel(H2Z, Z2H, DH, DZ)
DC.compile(optimizer=kr.optimizers.Adam(2e-5, .5),
           loss=kr.losses.SparseCategoricalCrossentropy(from_logits=True))

real_label = None
fake_label = None

GC_save_path = 'outputs/GC17.ckpt'
epoch_save_path = 'outputs/images17/epoch.npy'
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
            d_shape = DH(test_zebra).shape[1:-1]
            real_label = np.ones([batch_size, *d_shape])
            fake_label = np.zeros([batch_size, *d_shape])
            cv2.imshow('trainA&trainB', img3)
            cv2.waitKey()
            cv2.destroyAllWindows()
        cv2.imwrite('outputs/images17/trainA&z_%d.jpg' % epoch, img3)
    DC.fit([horse, zebra], [fake_label, fake_label, real_label, real_label], 6)
    np.random.shuffle(horse)
    GC.fit([horse, zebra], [real_label, real_label, horse, zebra], 6)
    if first:
        first = False
        GC.summary()
    try:
        GC.save_weights(GC_save_path)
    except Exception:
        pass
    epoch += 1
    np.save(epoch_save_path, epoch)
    print('Current epoch is %d' % epoch)
