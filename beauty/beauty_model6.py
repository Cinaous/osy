from abc import ABC
import tensorflow.keras as kr
from preprocessing_dataset3 import Dataset
import numpy as np
import os.path as path
from beauty.beauty_model9 import SplitDimsLayer


class MultiConv(kr.Model, ABC):
    def __init__(self):
        super(MultiConv, self).__init__()
        self.conv = [SplitDimsLayer(1 + i, 48, 5, 2) for i in range(6)]

    def call(self, inputs, training=None, mask=None):
        x = inputs
        for conv in self.conv:
            x = conv(x)
        return x


model = kr.Sequential([
    MultiConv(),
    kr.layers.Flatten(),
    kr.layers.Dense(2)
])
model_save_path = 'beauty6.ckpt'

if __name__ == '__main__':
    dataset = Dataset(batch_size=12)
    beauties, label_0, normals, label_1 = dataset.load_test()
    x_test = np.vstack([beauties, normals])
    y_test = np.hstack([label_0, label_1])
    model.compile(optimizer=kr.optimizers.Adam(2e-4),
                  loss=kr.losses.SparseCategoricalCrossentropy(True),
                  metrics=[kr.metrics.sparse_categorical_accuracy])
    model.build([None, *x_test.shape[1:]])
    model.summary()
    # if path.exists(f'{model_save_path}.index'):
    #     model.load_weights(model_save_path)
    cp = kr.callbacks.ModelCheckpoint(model_save_path,
                                      save_weights_only=True)
    for beauties, label_0, normals, label_1 in dataset:
        x_train = np.vstack([beauties, normals])
        y_train = np.hstack([label_0, label_1])
        model.fit(x_train, y_train, validation_data=(x_test, y_test),
                  callbacks=cp, workers=12, use_multiprocessing=True)
