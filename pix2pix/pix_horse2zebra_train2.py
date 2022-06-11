import abc
import cv2
import numpy as np
from dataset_horse2zebra import Dataset
from pix_models import Pix2PixUNet, Pix2PixEncoder
import tensorflow.keras as kr
import os.path as path
from pix_scaler import M3Scaler as Scaler

A2B = Pix2PixUNet()
B2A = Pix2PixUNet()
DA = Pix2PixEncoder()
DB = Pix2PixEncoder()


class CycleModel(kr.Model, abc.ABC):
    def __init__(self):
        super(CycleModel, self).__init__()
        self.a2b = A2B
        self.b2a = B2A

    def call(self, inputs, training=None, mask=None):
        a, b = inputs
        a2b = self.a2b(a)
        b2a = self.b2a(b)
        return self.b2a(a2b), self.a2b(b2a), DB(a2b), DA(b2a)


class DiscriminatorModel(kr.Model, abc.ABC):
    def __init__(self):
        super(DiscriminatorModel, self).__init__()
        self.da = DA
        self.db = DB

    def call(self, inputs, training=None, mask=None):
        a, b = inputs
        return self.da(a), self.db(b)


lr = 2e-4
cycleModel = CycleModel()
cycleModel.compile(optimizer=kr.optimizers.Adam(lr),
                   loss=[kr.losses.mae, kr.losses.mae,
                         kr.losses.mse, kr.losses.mse],
                   loss_weights=[10, 10, 1, 1])
discriminatorModel = DiscriminatorModel()
discriminatorModel.compile(optimizer=kr.optimizers.Adam(lr),
                           loss=kr.losses.mse)
cycle_save_path = './models/cycle2.ckpt'
discriminator_save_path = './models/discriminator2.ckpt'
if path.exists(cycle_save_path + '.index'):
    cycleModel.load_weights(cycle_save_path)
if path.exists(discriminator_save_path + '.index'):
    discriminatorModel.load_weights(discriminator_save_path)

dataset = Dataset()
scaler = Scaler()

epoch = 0
train_data_save_path = './images/horse2zebra2'
for trainA, trainB in dataset:
    if epoch % 25 == 0:
        testA, testB = dataset.load_test_data()
        test_a2b = A2B(scaler.fit_transform(testA))
        test_a2b2a = B2A(test_a2b)
        test_a2b, test_a2b2a = scaler.inverse_transform(test_a2b), scaler.inverse_transform(test_a2b2a)
        image_A = np.hstack([testA[0], test_a2b[0], test_a2b2a[0]])
        test_b2a = B2A(scaler.fit_transform(testB))
        test_b2a2b = A2B(test_b2a)
        test_b2a, test_b2a2b = scaler.inverse_transform(test_b2a), scaler.inverse_transform(test_b2a2b)
        image_B = np.hstack([testB[0], test_b2a[0], test_b2a2b[0]])
        image_AB = np.vstack([image_A, image_B])
        cv2.imwrite(f'{train_data_save_path}/{epoch}-train.jpg', image_AB)
    epoch += 1
    print(f'current epoch is {epoch}')
    trainA, trainB = scaler.fit_transform(trainA), scaler.fit_transform(trainB)
    real_label = np.ones(DA.compute_output_shape(trainA.shape))
    fake_label = np.zeros_like(real_label)
    discriminator_label = np.vstack([real_label, fake_label])
    discriminatorModel.fit([np.vstack([trainA, B2A(trainB)]),
                            np.vstack([trainB, A2B(trainA)])], [discriminator_label, discriminator_label])
    cycleModel.fit([trainA, trainB], [trainA, trainB, real_label, real_label])
    try:
        discriminatorModel.save_weights(discriminator_save_path)
        cycleModel.save_weights(cycle_save_path)
    except Exception as ex:
        print(ex)
