import os.path as path
import cv2
import numpy as np
import tensorflow as tf
import tensorflow.keras as kr
from horse2zebra_datasets9 import Datasets
from model_tree12 import Tree, TreeUNet
from cache_utils import Cache

batch_size = 1
dataset = Datasets(batch=batch_size)
H2Z = TreeUNet()
Z2H = TreeUNet()
DH = Tree()
DZ = Tree()
real_label = np.ones([batch_size, 14, 14, 32])
fake_label = np.zeros([batch_size, 14, 14, 32])
d_optimizer = kr.optimizers.Adam(2e-4)
z_optimizer = kr.optimizers.Adam(2e-4)
mae = kr.losses.MeanAbsoluteError()
dir_label = '73'
Z2H_save_path = 'outputs/Z2H%s.ckpt' % dir_label
H2Z_save_path = 'outputs/H2Z%s.ckpt' % dir_label
DH_save_path = 'outputs/DH%s.ckpt' % dir_label
DZ_save_path = 'outputs/DZ%s.ckpt' % dir_label
if path.exists(Z2H_save_path + '.index'):
    Z2H.load_weights(Z2H_save_path)
if path.exists(H2Z_save_path + '.index'):
    H2Z.load_weights(H2Z_save_path)
if path.exists(DH_save_path + '.index'):
    DH.load_weights(DH_save_path)
if path.exists(DZ_save_path + '.index'):
    DZ.load_weights(DZ_save_path)
epoch, first, h_cache, z_cache = 0, True, Cache(), Cache()
for horse, zebra in dataset:
    if epoch % 25 == 0 or first:
        test_horse, test_zebra = dataset.load_test_dataset(1)
        # test_horse = dataset.scaler.fit_transform(test_horse)
        h2z = H2Z(test_horse)
        h2z2h = Z2H(h2z)
        img1 = np.concatenate((test_horse, h2z, h2z2h), axis=2)[0]
        img1 = dataset.convert_data(img1)
        # test_zebra = dataset.scaler.fit_transform(test_zebra)
        z2h = Z2H(test_zebra)
        z2h2z = H2Z(z2h)
        img2 = np.concatenate((test_zebra, z2h, z2h2z), axis=2)[0]
        img2 = dataset.convert_data(img2)
        img3 = np.vstack((img1, img2))
        if first:
            first = False
            cv2.imshow('trainA&trainB', img3)
            cv2.waitKey()
            cv2.destroyAllWindows()
        cv2.imwrite('outputs/images%s/trainA&z_%d.jpg' % (dir_label, epoch), img3)
    with tf.GradientTape(watch_accessed_variables=False) as tape:
        tape.watch(H2Z.variables + Z2H.variables)
        h2z = H2Z(horse)
        loss = 10 * mae(horse, Z2H(h2z)) + mae(real_label, DZ(h2z))
        z2h = Z2H(zebra)
        loss += 10 * mae(zebra, H2Z(z2h)) + mae(real_label, DH(z2h))
    grads = tape.gradient(loss, tape.watched_variables())
    z_optimizer.apply_gradients(zip(grads, tape.watched_variables()))
    g_loss = loss
    with tf.GradientTape() as tape:
        loss = mae(real_label, DZ(z_cache(zebra))) + mae(fake_label, DZ(h2z))
        loss += mae(real_label, DH(h_cache(horse))) + mae(fake_label, DH(z2h))
    grads = tape.gradient(loss, tape.watched_variables())
    d_optimizer.apply_gradients(zip(grads, tape.watched_variables()))
    d_loss = loss
    try:
        H2Z.save_weights(H2Z_save_path)
        Z2H.save_weights(Z2H_save_path)
        DH.save_weights(DH_save_path)
        DZ.save_weights(DZ_save_path)
    except Exception as ex:
        print(ex.args)
    epoch += 1
    print('Current epoch is %d, g loss is %.5f, d loss is %.5f' % (epoch, g_loss, d_loss))
