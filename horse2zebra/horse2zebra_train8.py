from horse2zebra_datasets3 import Datasets
from models_model.model_models import UNet, Discriminator
import tensorflow as tf
import tensorflow.keras as kr
import os.path as path
import cv2
import numpy as np

dataset = Datasets()
train_horse, train_zebra = dataset.load_train_dataset()
test_horse, test_zebra = dataset.load_test_dataset()

H = UNet([64, 128, 256, 512, 512, 512, 512])
Z = UNet([64, 128, 256, 512, 512, 512, 512])
DH = Discriminator([128, 256, 512])
DZ = Discriminator([128, 256, 512])

d_optimizer = kr.optimizers.Adam(1e-6)
g_optimizer = kr.optimizers.Adam(1e-6)

H_save_path = 'outputs/H8.ckpt'
DH_save_path = 'outputs/DH8.ckpt'
Z_save_path = 'outputs/Z8.ckpt'
DZ_save_path = 'outputs/DZ8.ckpt'

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
        cv2.imwrite('outputs/test8_out_%d_%d.jpg' % (epoch, itr), img3)
    else:
        cv2.imshow('test', img3)
        cv2.waitKey(delay)
        cv2.destroyAllWindows()


bc = kr.losses.BinaryCrossentropy()
mae = kr.losses.MeanAbsoluteError()
test_show(save=False)

while True:
    itr = 0
    np.random.shuffle(train_horse)
    np.random.shuffle(train_zebra)
    for horse, zebra in zip(train_horse, train_zebra):
        horse, zebra = 2 * np.array([horse]), 2 * np.array([zebra])
        with tf.GradientTape(persistent=True) as tape:
            z2h = H(zebra, True)
            h2z = Z(horse, True)
            dh_fake = DH(np.concatenate((zebra, z2h), axis=-1), True)
            dh_real = DH(np.concatenate((zebra, horse), axis=-1), True)
            loss1 = bc(tf.zeros_like(dh_fake), dh_fake)
            loss2 = bc(tf.ones_like(dh_real), dh_real)

            dz_fake = DZ(np.concatenate((horse, h2z), axis=-1), True)
            dz_real = DZ(np.concatenate((horse, zebra), axis=-1), True)
            loss3 = bc(tf.zeros_like(dz_fake), dz_fake)
            loss4 = bc(tf.ones_like(dz_real), dz_real)

            loss_gan1 = bc(tf.ones_like(dh_fake), dh_fake)
            loss_gan2 = bc(tf.ones_like(dz_fake), dz_fake)

            z2h2z = Z(z2h, True)
            h2z2h = H(h2z, True)

            loss_gan_total = loss_gan1 + loss_gan2

            loss_cycle1 = mae(zebra, z2h2z)
            loss_cycle2 = mae(horse, h2z2h)
            loss_cycle_total = loss_cycle1 + loss_cycle2

            loss_total = (loss1 + loss2 + loss3 + loss4) / 2.
            loss_cycle_full = loss_gan_total + LAMBDA * loss_cycle_total
        d_grads = tape.gradient(loss_total, DH.trainable_variables + DZ.trainable_variables)
        d_optimizer.apply_gradients(zip(d_grads, DH.trainable_variables + DZ.trainable_variables))

        g_grads = tape.gradient(loss_cycle_full, H.trainable_variables + Z.trainable_variables)
        g_optimizer.apply_gradients(zip(g_grads, H.trainable_variables + Z.trainable_variables))

        try:
            DH.save_weights(DH_save_path)
            DZ.save_weights(DZ_save_path)
            H.save_weights(H_save_path)
            Z.save_weights(Z_save_path)
        except Exception:
            pass
        print('Current epoch is %d, iter is %d, g loss is %f, d loss is %f' % (epoch, itr, loss_cycle_full, loss_total))
        if itr % 50 == 0:
            test_show(horse / 2, zebra / 2)
        itr += 1
    epoch += 1
