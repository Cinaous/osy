import abc
import cv2
import numpy as np
from pix_models import RestNet131, RestUNet131
from pix2pix_train import load_data
import tensorflow.keras as kr
import os.path as path
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import copy

A2B = RestUNet131(6, 96)
B2A = copy.deepcopy(A2B)
DA = RestNet131(6, 96)
DB = copy.deepcopy(DA)


class CycleModel(kr.Model, abc.ABC):
    def __init__(self):
        super(CycleModel, self).__init__()
        self.A2B = A2B
        self.B2A = B2A

    def call(self, inputs, training=None, mask=None):
        a, b = inputs
        a2b = self.A2B(a)
        b2a = self.B2A(b)
        return self.B2A(a2b), self.A2B(b2a), DA(b2a), DB(a2b)


class DisModel(kr.Model, abc.ABC):
    def __init__(self):
        super(DisModel, self).__init__()
        self.DA = DA
        self.DB = DB

    def call(self, inputs, training=None, mask=None):
        a, b, a2b, b2a = inputs
        return self.DA(a), self.DB(b), self.DA(b2a), self.DB(a2b)


generator = ImageDataGenerator(
    rescale=1. / 255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True)
horse, zebra = [load_data(f'../horse2zebra/faces', label, (64, 64)) for label in ('trainA', 'trainB')]
lr, label, lambdas = 2e-4, 'face2face6', [10., 10., 1., 1.]
cycleModel = CycleModel()
cycleModel.compile(optimizer=kr.optimizers.Adam(lr),
                   loss=[kr.losses.mae, kr.losses.mae, kr.losses.mse, kr.losses.mse],
                   loss_weights=lambdas)
disModel = DisModel()
disModel.compile(optimizer=kr.optimizers.Adam(lr), loss=[kr.losses.mse, kr.losses.mse, kr.losses.mse, kr.losses.mse])
models = [cycleModel, disModel]
models_save_path = [f'./models/cycle-{label}-{i + 1}' for i in range(len(models))]
for model, model_save_path in zip(models, models_save_path):
    if path.exists(model_save_path + '.index'):
        model.load_weights(model_save_path)

real_label, fake_label, epoch = None, None, 0
train_data_save_path = f'./images/{label}'
for trainA, trainB in zip(generator.flow(horse), generator.flow(zebra)):
    if epoch == 0:
        real_label = np.ones(DA.compute_output_shape(trainA.shape))
        fake_label = np.zeros_like(real_label)
    if epoch % 25 == 0:
        testA, testB = trainA[:1], trainB[-1:]
        test_a2b = A2B(testA)
        test_a2b2a = B2A(test_a2b)
        image_A = np.hstack([testA[0], test_a2b[0], test_a2b2a[0]])
        test_b2a = B2A(testB)
        test_b2a2b = A2B(test_b2a)
        image_B = np.hstack([testB[0], test_b2a[0], test_b2a2b[0]])
        image_AB = (np.vstack([image_A, image_B]) * 255.).astype(np.uint8)
        cv2.imwrite(f'{train_data_save_path}/{epoch}-train.jpg', image_AB)
    epoch += 1
    print(f'current epoch is {epoch}: {trainA.shape}')
    disModel.fit([trainA, trainB, A2B(trainA), B2A(trainB)],
                 [real_label, real_label, fake_label, fake_label], 1,
                 workers=12, use_multiprocessing=True)
    cycleModel.fit([trainA, trainB], [trainA, trainB, real_label, real_label], 1,
                   workers=12, use_multiprocessing=True)
    try:
        [model.save_weights(model_save_path) for model, model_save_path in zip(models, models_save_path)]
    except Exception as ex:
        print(ex)
