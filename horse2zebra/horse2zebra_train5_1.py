from horse2zebra_datasets3 import Datasets
from GAN.bma.ma_model4 import edcoder, disc, dis
import tensorflow as tf
import tensorflow.keras as kr
import os.path as path
import cv2
import numpy as np

dataset = Datasets()
train_db = dataset.train_db(batch_size=3)
test_horse, test_zebra = dataset.load_test_dataset()
ns = np.log2(test_horse.shape[1])
ns = int(ns)

Z = edcoder(ns)
DZ = dis(ns)

d_optimizer = kr.optimizers.Adam()
g_optimizer = kr.optimizers.Adam()

Z_save_path = 'outputs/Z5.ckpt'
DZ_save_path = 'outputs/DZ5.ckpt'

# if path.exists(Z_save_path + '.index'):
#     Z.load_weights(Z_save_path)
#     DZ.load_weights(DZ_save_path)

LAMBDA, epoch = 10., 0


def test_show(test_horse=test_horse, test_zebra=test_zebra, delay=0, save=True):
    np.random.seed(None)
    idx = np.random.randint(len(test_horse), size=1)
    test_h, test_z = test_horse[idx], test_zebra[idx]
    h2z = Z(test_h, False)
    z2z = Z(test_z, False)
    img1 = np.hstack((test_h[0], h2z[0]))
    img2 = np.hstack((test_z[0], z2z[0]))
    img3 = np.vstack((img1, img2))
    img3 = dataset.convert_data(img3)
    if save:
        cv2.imwrite('outputs/test5_out_%d_%d.jpg' % (epoch, itr), img3)
    else:
        cv2.imshow('test', img3)
        cv2.waitKey(delay)
        cv2.destroyAllWindows()


test_show(save=False)

while True:
    itr = 0
    for horse, zebra in train_db:
        with tf.GradientTape(persistent=True) as tape:
            h2z = Z(horse, True)
            dz_fake = DZ(h2z, True)
            dz_real = DZ(zebra, True)
            loss3 = kr.losses.sparse_categorical_crossentropy(tf.zeros_like(dz_fake[..., 0]), dz_fake)
            loss4 = kr.losses.sparse_categorical_crossentropy(tf.ones_like(dz_real[..., 0]), dz_real)

            loss_gan2 = kr.losses.sparse_categorical_crossentropy(tf.ones_like(dz_fake[..., 0]), dz_fake)

            loss_total = loss3 + loss4
            loss_total = tf.reduce_mean(loss_total)
            loss_cycle_full = tf.reduce_mean(loss_gan2)
        d_grads = tape.gradient(loss_total, DZ.trainable_variables)
        d_optimizer.apply_gradients(zip(d_grads, DZ.trainable_variables))
        DZ.save_weights(DZ_save_path)

        g_grads = tape.gradient(loss_cycle_full, Z.trainable_variables)
        g_optimizer.apply_gradients(zip(g_grads, Z.trainable_variables))
        Z.save_weights(Z_save_path)
        print('Current epoch is %d, iter is %d, g loss is %f, d loss is %f' % (epoch, itr, loss_cycle_full, loss_total))
        if epoch % 50 == 0:
            test_show(horse.numpy(), zebra.numpy(), save=False, delay=5000)
        itr += 1
    epoch += 1
