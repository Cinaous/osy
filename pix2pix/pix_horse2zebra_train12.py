import abc
import cv2
import numpy as np
from dataset_pix2pix import Dataset
from pix_models import RestNetModel, RestUNetModel
import tensorflow.keras as kr
import os.path as path
from pix_scaler import StandardScaler as Scaler
import copy

A2B = RestUNetModel((15, 3), 128, 3, last_layer=kr.layers.Conv2DTranspose(3, 1))
B2A = copy.deepcopy(A2B)
DA = RestNetModel((6, 3), 128, 3, last_layer=kr.layers.Conv2D(1, 1))
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
        return self.B2A(a2b), self.A2B(b2a), DA(b2a), DB(a2b), self.B2A(a), self.A2B(b)


class DisModel(kr.Model, abc.ABC):
    def __init__(self):
        super(DisModel, self).__init__()
        self.DA = DA
        self.DB = DB

    def call(self, inputs, training=None, mask=None):
        a, b, a2b, b2a = inputs
        return self.DA(a), self.DB(b), self.DA(b2a), self.DB(a2b), self.DA(b), self.DB(a)


lr, label, lambdas = 2e-4, 'horse2zebra12', [5., 5., 1., 1., 2.5, 2.5]
cycleModel = CycleModel()
cycleModel.compile(optimizer=kr.optimizers.Adam(lr),
                   loss=[kr.losses.mae, kr.losses.mae, kr.losses.mse, kr.losses.mse, kr.losses.mae, kr.losses.mae],
                   loss_weights=lambdas)
disModel = DisModel()
disModel.compile(optimizer=kr.optimizers.Adam(lr),
                 loss=kr.losses.mse)
models = [cycleModel, disModel]
models_save_path = [f'./models/cycle-{label}-{i + 1}' for i in range(len(models))]
for model, model_save_path in zip(models, models_save_path):
    if path.exists(model_save_path + '.index'):
        model.load_weights(model_save_path)

dataset = Dataset('../horse2zebra')
real_label, fake_label, epoch, scaler = None, None, 0, Scaler()
train_data_save_path = f'./images/{label}'
for trainA, trainB in dataset:
    if epoch == 0:
        real_label = np.ones(DA.compute_output_shape(trainA.shape))
        fake_label = np.zeros_like(real_label)
        B2A.build(trainB.shape)
        DA.summary(), B2A.summary()
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
    print(f'current epoch is {epoch}: {trainA.shape}')
    trainA, trainB = scaler.fit_transform(trainA), scaler.fit_transform(trainB)
    disModel.fit([trainA, trainB, A2B(trainA), B2A(trainB)],
                 [real_label, real_label, fake_label, fake_label, fake_label, fake_label],
                 workers=12, use_multiprocessing=True)
    cycleModel.fit([trainA, trainB], [trainA, trainB, real_label, real_label, trainA, trainB],
                   workers=12, use_multiprocessing=True)
    try:
        [model.save_weights(model_save_path) for model, model_save_path in zip(models, models_save_path)]
    except Exception as ex:
        print(ex)
