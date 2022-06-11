import abc
import tensorflow.keras as kr
from dataset_pix2pix import Dataset
from pix_models import RestNetModel, RestUNetModel
from pix_scaler import FixedScaler as Scaler
import numpy as np
import tensorflow as tf
import cv2
import os.path as path

GNet = RestUNetModel((18, 3), 96, 3, last_layer=kr.layers.Conv2DTranspose(3, 1))
DNet = RestNetModel((9, 3), 96, 3, last_layer=kr.layers.Conv2D(1, 1))


class DTrain(kr.Model, abc.ABC):
    def __init__(self):
        super(DTrain, self).__init__()
        self.D = DNet

    def call(self, inputs, training=None, mask=None):
        real, fake = inputs
        return self.D(real), self.D(fake)


def g_loss(y_true, y_pred):
    return kr.losses.mse(y_true, DNet(y_pred))


real_label, fake_label = None, None
dataset = Dataset('../horse2zebra')
scale = Scaler()
file_save_path = './images/gan'
DT = DTrain()
DT.compile(optimizer=kr.optimizers.Adam(),
           loss=kr.losses.mse)
GNet.compile(optimizer=kr.optimizers.Adam(),
             loss=g_loss)
mds = [GNet, DT]
models_save_path = [f'./models/{md.name}' for md in mds]
# for model, save_path in zip(mds, models_save_path):
#     if path.exists(save_path + '.index'):
#         model.load_weights(save_path)
epoch = 0
for trainA, trainB in dataset:
    if not epoch:
        real_label = np.ones(DNet.compute_output_shape(trainA.shape))
        fake_label = np.zeros_like(real_label)
    if epoch % 25 == 0:
        testA, testB = dataset.load_test_data()
        preB = GNet(scale.fit_transform(testA))
        preB = scale.inverse_transform(preB)
        img = np.hstack([testA[0], preB[0]])
        cv2.imwrite(f'{file_save_path}/gan-{epoch}.jpg', img)
    epoch += 1
    print(f"Current epoch is {epoch}")
    trainA, trainB = scale.fit_transform(trainA), scale.transform(trainB)
    DT.fit([trainB, GNet(trainA)], [real_label, fake_label], workers=12, use_multiprocessing=True)
    GNet.fit(trainA, real_label, workers=12, use_multiprocessing=True)
    try:
        [model.save_weights(save_path) for model, save_path in zip(mds, models_save_path)]
    except Exception as ex:
        print(ex)
