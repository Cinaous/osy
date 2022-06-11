import abc

from pix2pix.pix2pix_train import load_data
import numpy as np
from pix2pix.pix_models import RestNetPLayer, RestUNetPLayer
import tensorflow.keras as kr
import os.path as path


def load_train_data():
    horse = load_data('../horse2zebra/faces', 'trainA', (64, 64))
    horse_label = np.ones(len(horse))
    zebra = load_data('../horse2zebra/faces', 'trainB', (64, 64))
    zebra_label = np.zeros(len(zebra))
    return horse, zebra, horse_label, zebra_label


A2B = RestUNetPLayer((15, 3), 48, 3, last_layer=kr.layers.Conv2DTranspose(3, 3, padding='same'))
B2A = RestUNetPLayer((15, 3), 48, 3, last_layer=kr.layers.Conv2DTranspose(3, 3, padding='same'))
DC = RestNetPLayer((15, 3), 48, 3, last_layer=kr.Sequential([
    kr.layers.Flatten(),
    kr.layers.Dense(2)
]))


class CycleModel(kr.Model, abc.ABC):
    def __init__(self):
        super(CycleModel, self).__init__()
        self.A2B = A2B
        self.B2A = B2A

    def call(self, inputs, training=None, mask=None):
        a, b = inputs
        a2b, b2a = self.A2B(a), self.B2A(b)
        return self.A2B(a2b), self.B2A(b2a), DC(a2b), DC

if __name__ == '__main__':
    x_train, y_train = load_train_data()
    Net.fit(x_train, y_train, 64, 50, use_multiprocessing=True, workers=12, callbacks=[cp])
