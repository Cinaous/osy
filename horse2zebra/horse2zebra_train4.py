from horse2zebra_datasets3 import Datasets
from horse2zebra_models import PatchGAN, UNet
import tensorflow as tf
import tensorflow.keras as kr
import os.path as path
import cv2
import numpy as np

dataset = Datasets()
train_db = dataset.train_db(batch_size=8)
test_horse, test_zebra = dataset.load_test_dataset()

H = UNet()
Z = UNet()
DH = PatchGAN()
DZ = PatchGAN()

d_optimizer = kr.optimizers.Adam()
g_optimizer = kr.optimizers.Adam()

H_save_path = 'outputs/H2.ckpt'
DH_save_path = 'outputs/DH2.ckpt'
Z_save_path = 'outputs/Z2.ckpt'
DZ_save_path = 'outputs/DZ2.ckpt'

if path.exists(H_save_path + '.index'):
    H.load_weights(H_save_path)
    DH.load_weights(DH_save_path)
    Z.load_weights(Z_save_path)
    DZ.load_weights(DZ_save_path)

LAMBDA, epoch = 10., 0


def test_show(test_horse=test_horse, test_zebra=test_zebra, delay=0, save=True):
    np.random.seed(None)
    idx = np.random.randint(len(test_horse), size=1)
    test_h, test_z = test_horse[idx], test_zebra[idx]
    z2h = H(test_z, False)
    h2z = Z(test_h, False)
    z2z = Z(z2h, False)
    h2h = H(h2z, False)
    img1 = np.hstack((test_h[0], h2z[0], h2h[0]))
    img2 = np.hstack((test_z[0], z2h[0], z2z[0]))
    img3 = np.vstack((img1, img2))
    img3 = dataset.convert_data(img3)
    if save:
        cv2.imwrite('outputs/test4_out_%d.jpg' % epoch, img3)
    else:
        cv2.imshow('test', img3)
        cv2.waitKey(delay)
        cv2.destroyAllWindows()


test_show(save=False)

while True:
    for horse, zebra in train_db:
        with tf.GradientTape(persistent=True) as tape:
            z2h = H(zebra, True)
            h2z = Z(horse, True)
            dh_fake = DH(z2h, True)
            dh_real = DH(horse, True)
            loss1 = kr.losses.binary_crossentropy(tf.zeros_like(dh_fake), dh_fake)
            loss2 = kr.losses.binary_crossentropy(tf.ones_like(dh_real), dh_real)

            dz_fake = DZ(h2z, True)
            dz_real = DZ(zebra, True)
            loss3 = kr.losses.binary_crossentropy(tf.zeros_like(dz_fake), dz_fake)
            loss4 = kr.losses.binary_crossentropy(tf.ones_like(dz_real), dz_real)
            loss_total = loss1 + loss2 + loss3 + loss4
            loss_total = tf.reduce_mean(loss_total)

            loss_gan1 = kr.losses.binary_crossentropy(tf.ones_like(dh_fake), dh_fake)
            loss_gan2 = kr.losses.binary_crossentropy(tf.ones_like(dz_fake), dz_fake)
            loss_gan_total = loss_gan1 + loss_gan2
            loss_gan_total = tf.reduce_mean(loss_gan_total)

            z2h2z = Z(z2h, True)
            h2z2h = H(h2z, True)
            loss_cycle1 = kr.losses.mae(zebra, z2h2z)
            loss_cycle2 = kr.losses.mae(horse, h2z2h)
            loss_cycle_total = loss_cycle1 + loss_cycle2
            loss_cycle_total = tf.reduce_mean(loss_cycle_total)

            loss_cycle_full = loss_gan_total + LAMBDA * loss_cycle_total
        d_grads = tape.gradient(loss_total, DH.trainable_variables + DZ.trainable_variables)
        d_optimizer.apply_gradients(zip(d_grads, DH.trainable_variables + DZ.trainable_variables))
        DH.save_weights(DH_save_path)
        DZ.save_weights(DZ_save_path)

        g_grads = tape.gradient(loss_cycle_full, H.trainable_variables + Z.trainable_variables)
        g_optimizer.apply_gradients(zip(g_grads, H.trainable_variables + Z.trainable_variables))
        H.save_weights(H_save_path)
        Z.save_weights(Z_save_path)
        print('Current epoch is %d, g loss is %f, d loss is %f' % (epoch, loss_cycle_full, loss_total))
    if epoch % 50 == 0:
        test_show(horse.numpy(), zebra.numpy())
    epoch += 1
