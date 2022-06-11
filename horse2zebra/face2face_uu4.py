import os.path as path
from abc import ABC
import cv2
import numpy as np
import tensorflow as tf
import tensorflow.keras as kr
from face2face_datasets1 import Datasets
from models_model.model_models14 import UNet
import matplotlib.pyplot as plt
from cache_utils import Cache

batch_size = 1
dataset = Datasets(batch=batch_size)
Z2H = UNet([64, 128, 256, 512, 512], activation=kr.activations.tanh)
H2Z = UNet([64, 128, 256, 512, 512], activation=kr.activations.tanh)
DH = UNet([64, 128, 256, 512, 512], activation=kr.activations.sigmoid)
DZ = UNet([64, 128, 256, 512, 512], activation=kr.activations.sigmoid)


class StepsLearningRateDecay(kr.optimizers.schedules.LearningRateSchedule, ABC):
    def __init__(self, initial_learning_rate):
        self.initial_learning_rate = tf.Variable(initial_learning_rate, False, dtype=tf.float32)

    def __call__(self, step):
        self.initial_learning_rate.assign(
            tf.cond(step % 500 != 0,
                    true_fn=lambda: self.initial_learning_rate,
                    false_fn=lambda:
                    tf.cond(step < 10000,
                            true_fn=lambda: self.initial_learning_rate * .99,
                            false_fn=lambda: self.initial_learning_rate * .96)))
        return self.initial_learning_rate


real_label = None
fake_label = None

dir_label = '45'
DH_save_path = 'outputs/GH%s.ckpt' % dir_label
DZ_save_path = 'outputs/DZ%s.ckpt' % dir_label
Z2H_save_path = 'outputs/Z2H%s.ckpt' % dir_label
H2Z_save_path = 'outputs/H2Z%s.ckpt' % dir_label
epoch_save_path = 'outputs/images%s/epoch.npy' % dir_label
loss_save_path = 'outputs/images%s/loss.npy' % dir_label
epoch, first, losses, init_lr = 0, True, [], None
if path.exists(epoch_save_path):
    epoch, init_lr = np.load(epoch_save_path).tolist()
    losses = np.load(loss_save_path)
    er = int(epoch // 100)
    ls = [np.mean(losses[i * er:(i + 1) * er], 0) for i in range(100)]
    ls = np.array(ls)
    plt.plot(ls[:, 0], label='g_loss')
    plt.plot(ls[:, 1], label='d_loss')
    losses = losses.tolist()
    plt.legend()
    plt.title('loss')
    plt.show()

init_lr = 1e-3 if init_lr is None else init_lr
g_schedue = StepsLearningRateDecay(init_lr)
d_schedue = StepsLearningRateDecay(init_lr)
g_optimizer = kr.optimizers.Adam(g_schedue, .5)
d_optimizer = kr.optimizers.Adam(d_schedue, .5)
mse = kr.losses.MeanSquaredError()
cycle_lambda, z_cache, h_cache = 5, Cache(), Cache()

if path.exists(DZ_save_path + '.index'):
    DZ.load_weights(DZ_save_path)
if path.exists(DH_save_path + '.index'):
    DH.load_weights(DH_save_path)
if path.exists(Z2H_save_path + '.index'):
    Z2H.load_weights(Z2H_save_path)
if path.exists(H2Z_save_path + '.index'):
    H2Z.load_weights(H2Z_save_path)

for horse, zebra in dataset:
    if epoch % 25 == 0 or first:
        test_horse, test_zebra = horse[-1:], zebra[-1:]
        H2Z.trainable, DH.trainable = False, False
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
            d_shape = DH(test_zebra).shape[1:]
            H2Z.summary()
            DH.summary()
            real_label = np.ones([batch_size, *d_shape])
            fake_label = np.zeros([batch_size, *d_shape])
            cv2.imshow('trainA&trainB', img3)
            cv2.waitKey()
            cv2.destroyAllWindows()
        cv2.imwrite('outputs/images%s/trainA&z_%d.jpg' % (dir_label, epoch), img3)
    # train G
    DH.trainable, DZ.trainable, H2Z.trainable, Z2H.trainable = False, False, True, True
    with tf.GradientTape() as tape:
        h2z = H2Z(horse)
        h2z2h = Z2H(h2z)
        loss = mse(real_label, DZ(h2z)) + cycle_lambda * mse(horse, h2z2h)
        z2h = Z2H(zebra)
        z2h2z = H2Z(z2h)
        loss += mse(real_label, DH(z2h)) + cycle_lambda * mse(zebra, z2h2z)
    grads = tape.gradient(loss, H2Z.trainable_variables + Z2H.trainable_variables)
    g_optimizer.apply_gradients(zip(grads, H2Z.trainable_variables + Z2H.trainable_variables))
    g_loss = loss.numpy()

    # train D
    DH.trainable, DZ.trainable, H2Z.trainable, Z2H.trainable = True, True, False, False
    with tf.GradientTape() as tape:
        loss = mse(real_label, DZ(zebra)) + mse(fake_label, DZ(z_cache(h2z)))
        loss += mse(real_label, DH(horse)) + mse(fake_label, DH(h_cache(z2h)))
    grads = tape.gradient(loss, DZ.trainable_variables + DH.trainable_variables)
    d_optimizer.apply_gradients(zip(grads, DZ.trainable_variables + DH.trainable_variables))
    d_loss = loss.numpy()

    try:
        Z2H.save_weights(Z2H_save_path)
        H2Z.save_weights(H2Z_save_path)
        DH.save_weights(DH_save_path)
        DZ.save_weights(DZ_save_path)
    except Exception:
        pass
    losses.append([g_loss, d_loss])
    epoch += 1
    lr = g_optimizer.lr.initial_learning_rate.numpy()
    np.save(loss_save_path, losses)
    np.save(epoch_save_path, [epoch, lr])
    print('current epoch is %d: \n\tg is %.5f, d is %.5f, lr is %.8f' % (epoch, g_loss, d_loss, lr))
