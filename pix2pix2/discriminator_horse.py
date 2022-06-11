from pix2pix.pix2pix_train import load_data
import numpy as np
from pix2pix.pix_models import RestNetPLayer
import tensorflow.keras as kr
import os.path as path


def load_train_data():
    horse = load_data('../horse2zebra/faces', 'trainA', (64, 64))
    horse_label = np.ones(len(horse))
    zebra = load_data('../horse2zebra/faces', 'trainB', (64, 64))
    zebra_label = np.zeros(len(zebra))
    x_train = np.vstack((horse, zebra)) / 255.
    y_train = np.concatenate((horse_label, zebra_label))
    return x_train, y_train


Net = RestNetPLayer((15, 3), 48, 3, last_layer=kr.Sequential([
    kr.layers.Flatten(),
    kr.layers.Dense(2)
]))
save_path = 'mds/dis_horse.ckpt'
if path.exists(save_path + '.index'):
    Net.load_weights(save_path)
Net.compile(optimizer=kr.optimizers.Adam(),
            loss=kr.losses.SparseCategoricalCrossentropy(True),
            metrics=kr.metrics.sparse_categorical_accuracy)
cp = kr.callbacks.ModelCheckpoint(save_path, save_weights_only=True,
                                  save_best_only=True,
                                  monitor='loss')

if __name__ == '__main__':
    x_train, y_train = load_train_data()
    Net.fit(x_train, y_train, 64, 50, use_multiprocessing=True, workers=12, callbacks=[cp])
