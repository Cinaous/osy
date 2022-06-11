import abc
import cv2
import numpy as np
from dataset_pix2pix import Dataset
from pix_models import RestUModel, RestModel
import tensorflow.keras as kr
import os.path as path
from pix_scaler import M3Scaler as Scaler

A2B = RestUModel([5 if i != 4 else 1 for i in range(5)], 96, 3, last_layer=kr.layers.Dense(3, kr.activations.tanh))
B2A = RestUModel([5 if i != 4 else 1 for i in range(5)], 96, 3, last_layer=kr.layers.Dense(3, kr.activations.tanh))
DA = RestModel([4 for _ in range(3)], 96, 3, last_layer=kr.layers.Dense(1, kr.activations.sigmoid))
DB = RestModel([4 for _ in range(3)], 96, 3, last_layer=kr.layers.Dense(1, kr.activations.sigmoid))


class CycleModel(kr.Model, abc.ABC):
    def __init__(self):
        super(CycleModel, self).__init__()
        self.a2b = A2B
        self.b2a = B2A

    def call(self, inputs, training=None, mask=None):
        a, b = inputs
        a2b = self.a2b(a)
        b2a = self.b2a(b)
        return self.b2a(a2b), self.a2b(b2a), DB(a2b), DA(b2a), self.b2a(a), self.a2b(b)


class DiscriminatorModel(kr.Model, abc.ABC):
    def __init__(self):
        super(DiscriminatorModel, self).__init__()
        self.da = DA
        self.db = DB

    def call(self, inputs, training=None, mask=None):
        a, b = inputs
        return self.da(a), self.db(b), self.da(B2A(b)), self.db(A2B(a))


lr = 2e-4
cycleModel = CycleModel()
cycleModel.compile(optimizer=kr.optimizers.Adam(lr),
                   loss=[kr.losses.mae, kr.losses.mae,
                         kr.losses.mse, kr.losses.mse,
                         kr.losses.mae, kr.losses.mae],
                   loss_weights=[5, 5, 1, 1, 2.5, 2.5])
discriminatorModel = DiscriminatorModel()
discriminatorModel.compile(optimizer=kr.optimizers.Adam(lr),
                           loss=kr.losses.mse)
cycle_save_path = './models/face2face3_cycle.ckpt'
discriminator_save_path = './models/face2face3_discriminator.ckpt'
if path.exists(cycle_save_path + '.index'):
    cycleModel.load_weights(cycle_save_path)
if path.exists(discriminator_save_path + '.index'):
    discriminatorModel.load_weights(discriminator_save_path)

dataset = Dataset('../horse2zebra/faces', 32)
scaler = Scaler()
real_label = None
fake_label = None

epoch = 0
train_data_save_path = './images/face2face3'
for trainA, trainB in dataset:
    real_label = np.ones(DA.compute_output_shape(trainA.shape)) if real_label is None else real_label
    fake_label = np.zeros_like(real_label) if fake_label is None else fake_label
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
    try:
        discriminatorModel.fit([trainA, trainB], [real_label, real_label, fake_label, fake_label])
        cycleModel.fit([trainA, trainB], [trainA, trainB, real_label, real_label, trainA, trainB])
        discriminatorModel.save_weights(discriminator_save_path)
        cycleModel.save_weights(cycle_save_path)
    except Exception as ex:
        print(ex)
