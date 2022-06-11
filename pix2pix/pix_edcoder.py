import abc
import os.path as path
import cv2
import numpy as np
import tensorflow.keras as kr
from dataset_horse2zebra import Dataset
from pix_models import Pix2PixEncoder, Pix2PixDecoder
from pix_scaler import M3Scaler as Scaler


class AutoCoder(kr.Model, abc.ABC):
    def __init__(self):
        super(AutoCoder, self).__init__()
        self.E = Pix2PixEncoder()
        self.DA = Pix2PixDecoder()
        self.DB = Pix2PixDecoder()

    def call(self, inputs, training=None, mask=None):
        a, b = inputs
        return self.DA(self.E(a)), self.DB(self.E(b))

    def pix2pix(self, inputs):
        a, b = inputs
        return self.DA(self.E(b)), self.DB(self.E(a))


model = AutoCoder()
model.compile(optimizer=kr.optimizers.Adam(), loss=kr.losses.mae)
model_save_path = './models/ac1.ckpt'
if path.exists(model_save_path + '.index'):
    model.load_weights(model_save_path)

scaler_A, scaler_B = Scaler(), Scaler()
dataset = Dataset()
epoch = 0
train_data_save_path = './images/ac1'
for trainA, trainB in dataset:
    if epoch % 25 == 0:
        testA, testB = dataset.load_test_data()
        b2a, a2b = model.pix2pix([scaler_A.fit_transform(testA), scaler_B.fit_transform(testB)])
        a2b, b2a = scaler_A.inverse_transform(a2b), scaler_B.inverse_transform(b2a)
        image_AB = np.hstack([testA[0], a2b[0], testB[0], b2a[0]])
        cv2.imwrite(f'{train_data_save_path}/{epoch}-train.jpg', image_AB)
    epoch += 1
    print(f'Current epoch is {epoch}:')
    trainA, trainB = scaler_A.fit_transform(trainA), scaler_B.fit_transform(trainB)
    model.fit([trainA, trainB], [trainA, trainB])
    try:
        model.save_weights(model_save_path)
    except Exception as ex:
        print(ex)
