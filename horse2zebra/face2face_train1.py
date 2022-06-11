import os
import os.path as path
import cv2
import numpy as np
import tensorflow.keras as kr
import tensorflow as tf
from face2face_datasets1 import Datasets
from models_model.model_models7 import UNet, Discriminator, DCModel, GCModel

threads = os.cpu_count() // 3
tf.config.threading.set_intra_op_parallelism_threads(threads)
tf.config.threading.set_inter_op_parallelism_threads(threads)

batch_size = 50
dataset = Datasets(batch=batch_size)
Z2H = UNet([64, 128, 256, 512, 512, 512, 512])
H2Z = UNet([64, 128, 256, 512, 512, 512, 512])
DH = Discriminator([64, 128, 256, 512])
DZ = Discriminator([64, 128, 256, 512])

GC = GCModel(H2Z, Z2H, DH, DZ)
GC.compile(optimizer=kr.optimizers.Adam(),
           loss=GCModel.loss_cycle)
DC = DCModel(H2Z, Z2H, DH, DZ)
DC.compile(optimizer=kr.optimizers.Adam(),
           loss=kr.losses.sparse_categorical_crossentropy)

real_label = None
fake_label = None

DC_save_path = 'outputs/DC4.ckpt'
DC_callback = kr.callbacks.ModelCheckpoint(DC_save_path, save_weights_only=True, monitor='loss')
GC_save_path = 'outputs/GC4.ckpt'
GC_callback = kr.callbacks.ModelCheckpoint(GC_save_path, save_weights_only=True, monitor='loss')
if path.exists(DC_save_path + '.index'):
    DC.load_weights(DC_save_path)
    GC.load_weights(GC_save_path)

epoch_save_path = 'outputs/images4/epoch.npy'
epoch, first = 0, True
if path.exists(epoch_save_path):
    epoch = np.load(epoch_save_path).max()

for horse, zebra in dataset:
    if epoch % 2 == 0 or first:
        test_horse, test_zebra = dataset.load_test_dataset(1)
        test_horse_ = np.concatenate((test_horse, test_zebra), axis=-1)
        test_zebra_ = np.concatenate((test_zebra, test_horse), axis=-1)
        h2z = H2Z(test_horse_)
        h2z2h = Z2H(np.concatenate((h2z, test_horse), axis=-1))
        img1 = np.concatenate((test_horse, h2z, h2z2h), axis=2)[0]
        z2h = Z2H(test_zebra_)
        z2h2z = H2Z(np.concatenate((z2h, test_zebra), axis=-1))
        img2 = np.concatenate((test_zebra, z2h, z2h2z), axis=2)[0]
        img3 = np.vstack((img1, img2))
        img3 = Datasets.convert_data(img3)
        if first:
            first = False
            d_shape = DH(test_zebra_).shape[1:-1]
            real_label = np.ones([batch_size, *d_shape])
            fake_label = np.zeros([batch_size, *d_shape])
            cv2.imshow('trainA&trainB', img3)
            cv2.waitKey()
            cv2.destroyAllWindows()
        cv2.imwrite('outputs/images4/trainA&z_%d.jpg' % epoch, img3)
    DC.fit([horse, zebra], [fake_label, fake_label, real_label, real_label], batch_size=1, callbacks=[DC_callback])
    DC.summary()
    GC.fit([horse, zebra], [real_label, real_label, horse, zebra, horse, zebra], batch_size=1, callbacks=[GC_callback])
    GC.summary()
    epoch += 1
    np.save(epoch_save_path, epoch)
    print('Current epoch is %d' % epoch)
