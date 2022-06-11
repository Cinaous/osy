from horse2zebra_datasets4 import Datasets
from models_model.model_models3 import UNet, Discriminator
import tensorflow as tf
import tensorflow.keras as kr
import os.path as path
import cv2
import numpy as np

dataset = Datasets()
test_horse, test_zebra = dataset.load_test_dataset()

H = UNet([64, 128, 256, 512, 512, 512, 512])
Z = UNet([64, 128, 256, 512, 512, 512, 512])
DH = Discriminator([64, 128, 256, 512])
DZ = Discriminator([64, 128, 256, 512])

d_optimizer = kr.optimizers.Adam()
g_optimizer = kr.optimizers.Adam()

H_save_path = 'outputs/H12.ckpt'
DH_save_path = 'outputs/DH12.ckpt'
Z_save_path = 'outputs/Z12.ckpt'
DZ_save_path = 'outputs/DZ12.ckpt'

if path.exists(H_save_path + '.index'):
    H.load_weights(H_save_path)
    DH.load_weights(DH_save_path)
    Z.load_weights(Z_save_path)
    DZ.load_weights(DZ_save_path)

LAMBDA, epoch = 1., 0


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
        cv2.imwrite('outputs/test12_out11_%d.jpg' % epoch, img3)
    else:
        cv2.imshow('test', img3)
        cv2.waitKey(delay)
        cv2.destroyAllWindows()


scc = kr.losses.SparseCategoricalCrossentropy(from_logits=True)
mse = kr.losses.MeanAbsoluteError()
test_show(save=False)

real_label = fake_label = None

for horse, zebra in dataset:
    with tf.GradientTape(persistent=True) as tape:
        z2h = H(zebra, True)
        h2z = Z(horse, True)
        dh_fake = DH(z2h, True)
        dh_real = DH(horse, True)
        if real_label is None:
            real_label = np.ones(dh_real.shape[:-1])
            fake_label = np.zeros(dh_fake.shape[:-1])
        loss1 = scc(fake_label, dh_fake)
        loss2 = scc(real_label, dh_real)

        dz_fake = DZ(h2z, True)
        dz_real = DZ(zebra, True)
        loss3 = scc(fake_label, dz_fake)
        loss4 = scc(real_label, dz_real)

        loss_gan1 = scc(real_label, dh_fake)
        loss_gan2 = scc(real_label, dz_fake)
        loss_gan_total = loss_gan1 + loss_gan2

        z2h2z = Z(z2h, True)
        h2z2h = H(h2z, True)

        loss_cycle1 = mse(zebra, z2h2z)
        loss_cycle2 = mse(horse, h2z2h)
        loss_cycle_total = loss_cycle1 + loss_cycle2

        # z2z = Z(zebra, True)
        # loss_idt1 = mse(zebra, z2z)
        # h2h = H(horse, True)
        # loss_idt2 = mse(horse, h2h)
        # loss_idt_total = loss_idt1 + loss_idt2
        loss_idt_total = 0

        loss_total = loss1 + loss2 + loss3 + loss4
        loss_cycle_full = loss_gan_total + LAMBDA * loss_cycle_total + .5 * LAMBDA * loss_idt_total
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
    print('Current epoch is %d, g loss is %f, cycle loss is %f, d loss is %f, i loss is %f' %
          (epoch, loss_gan_total, loss_cycle_total, loss_total, loss_idt_total))
    if epoch % 50 == 0:
        test_show(horse, zebra)
    epoch += 1
