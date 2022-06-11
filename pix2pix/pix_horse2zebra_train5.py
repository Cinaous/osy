import abc
import cv2
import numpy as np
from dataset_horse2zebra import Dataset, resize_data as resize_train_data
from pix_models import ConvUNet, ConvModel
import tensorflow.keras as kr
import os.path as path
from pix_scaler import M3Scaler as Scaler
import copy
from cache_utils import Cache

A2B = ConvUNet(96, 5, 6)
B2A = copy.deepcopy(A2B)
DA = ConvModel(96, 5, 3)
DB = copy.deepcopy(DA)


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
        a, b, a_, b_ = inputs
        return self.da(a), self.db(b), self.da(a_), self.db(b_)


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
cycle_save_path = './models/cycle5.ckpt'
discriminator_save_path = './models/discriminator5.ckpt'
if path.exists(cycle_save_path + '.index'):
    cycleModel.load_weights(cycle_save_path)
if path.exists(discriminator_save_path + '.index'):
    discriminatorModel.load_weights(discriminator_save_path)

dataset = Dataset(resize=False)
scaler = Scaler()

epoch = 0
train_data_save_path = './images/horse2zebra5'
cacheA, cacheB = Cache(), Cache()
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
    scale_width = np.random.randint(1, 5) * 64
    trainA, trainB = resize_train_data(trainA, scale_width), resize_train_data(trainB, scale_width)
    if np.random.uniform() > .6 or cacheA.batch_size is None:
        a2b, b2a = cacheB(A2B(trainA)), cacheA(B2A(trainB))
    else:
        a2b, b2a = cacheB.sample(), cacheA.sample()
    a2b, b2a = resize_train_data(a2b, scale_width), resize_train_data(b2a, scale_width)
    real_label = np.ones(DA.compute_output_shape(trainA.shape))
    fake_label = np.zeros_like(real_label)
    discriminatorModel.fit([trainA, trainB, b2a, a2b], [real_label, real_label, fake_label, fake_label])
    cycleModel.fit([trainA, trainB], [trainA, trainB, real_label, real_label, trainA, trainB])
    try:
        discriminatorModel.save_weights(discriminator_save_path)
        cycleModel.save_weights(cycle_save_path)
    except Exception as ex:
        print(ex)
