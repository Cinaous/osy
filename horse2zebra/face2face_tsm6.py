import os.path as path
from abc import ABC
import cv2
import numpy as np
import os
import tensorflow as tf
import tensorflow.keras as kr
from face2face_datasets1 import Datasets
from trm.swin_transformer11 import SwinTransformerBlock
from models_model.model_models12 import DCModel, GCModel
import matplotlib.pyplot as plt

threads = os.cpu_count() // 2
tf.config.threading.set_intra_op_parallelism_threads(threads)
tf.config.threading.set_inter_op_parallelism_threads(threads)

batch_size = 1
dataset = Datasets(batch=batch_size)
Z2H = kr.Sequential([
    SwinTransformerBlock(128),
    SwinTransformerBlock(128)
])
H2Z = kr.Sequential([
    SwinTransformerBlock(128),
    SwinTransformerBlock(128)
])
DH = kr.Sequential([
    SwinTransformerBlock(128),
    SwinTransformerBlock(64)
])
DZ = kr.Sequential([
    SwinTransformerBlock(128),
    SwinTransformerBlock(64)
])


class StepsLearningRateDecay(kr.optimizers.schedules.LearningRateSchedule, ABC):
    def __init__(self, initial_learning_rate):
        self.initial_learning_rate = tf.Variable(initial_learning_rate, False, dtype=tf.float32)

    def __call__(self, step):
        self.initial_learning_rate.assign(
            tf.cond(step % 1000 != 0,
                    true_fn=lambda: self.initial_learning_rate,
                    false_fn=lambda:
                    tf.cond(step < 30000,
                            true_fn=lambda: self.initial_learning_rate,
                            false_fn=lambda: self.initial_learning_rate * .96)))
        return self.initial_learning_rate


real_label = None
fake_label = None

GC_save_path = 'outputs/GC28.ckpt'
DC_save_path = 'outputs/DC28.ckpt'
epoch_save_path = 'outputs/images28/epoch.npy'
loss_save_path = 'outputs/images28/loss.npy'
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

init_lr = 2e-4 if init_lr is None else init_lr
g_schedue = StepsLearningRateDecay(init_lr)
d_schedue = StepsLearningRateDecay(init_lr)
GC = GCModel(H2Z, Z2H, DH, DZ)
GC.compile(optimizer=kr.optimizers.Adam(g_schedue, .5),
           loss=[kr.losses.mse, kr.losses.mse,
                 kr.losses.mae, kr.losses.mae,
                 kr.losses.mae, kr.losses.mae],
           loss_weights=[1, 1, 10, 10, 5, 5])
DC = DCModel(H2Z, Z2H, DH, DZ)
DC.compile(optimizer=kr.optimizers.Adam(d_schedue, .5),
           loss=kr.losses.mse)

if path.exists(DC_save_path + '.index'):
    print('load GC weights...')
    GC.load_weights(GC_save_path)
    DC.load_weights(DC_save_path)

for horse, zebra in dataset:
    if epoch % 100 == 0 or first:
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
            d_shape = DH(test_zebra).shape[1:]
            real_label = np.ones([batch_size, *d_shape])
            fake_label = np.zeros([batch_size, *d_shape])
            cv2.imshow('trainA&trainB', img3)
            cv2.waitKey()
            cv2.destroyAllWindows()
        cv2.imwrite('outputs/images28/trainA&z_%d.jpg' % epoch, img3)
    g_loss = GC.train_on_batch([horse, zebra], [real_label, real_label, horse, zebra, horse, zebra])[0]
    d_loss = DC.train_on_batch([horse, zebra], [fake_label, fake_label, real_label, real_label])[0]
    if first:
        first = False
        H2Z.summary()
        DH.summary()
    try:
        GC.save_weights(GC_save_path)
        DC.save_weights(DC_save_path)
    except Exception:
        pass
    losses.append([g_loss, d_loss])
    epoch += 1
    lr = GC.optimizer.lr.initial_learning_rate.numpy()
    np.save(loss_save_path, losses)
    np.save(epoch_save_path, [epoch, lr])
    print('current epoch is %d: \n\tg is %.5f, d is %.5f, lr is %.8f' % (epoch, g_loss, d_loss, lr))
