from abc import ABC
import tensorflow.keras as kr
from preprocessing_dataset3 import Dataset
import numpy as np
import tensorflow as tf
import os.path as path

activation = kr.layers.Activation(kr.activations.swish)


class Model(kr.Model, ABC):
    def __init__(self):
        super(Model, self).__init__()
        self.embed = kr.layers.Embedding(256, 16)
        # 64
        self.conv1 = kr.Sequential([
            kr.layers.Conv2D(48, 11, 2, 'same'),
            kr.layers.LayerNormalization(),
            activation
        ])
        # 56
        self.conv2 = kr.Sequential([
            kr.layers.Conv2D(56, 9),
            kr.layers.LayerNormalization(),
            activation,
            kr.layers.Dropout(.2)
        ])
        # 28
        self.conv3 = kr.Sequential([
            kr.layers.Conv2D(64, 7, 2, 'same'),
            kr.layers.LayerNormalization(),
            activation
        ])
        # 14
        self.conv4 = kr.Sequential([
            kr.layers.Conv2D(86, 5, 2, 'same'),
            kr.layers.LayerNormalization(),
            activation
        ])
        # 14
        self.conv5 = kr.Sequential([
            kr.layers.Conv2D(32, 9, 4, 'same'),
            kr.layers.LayerNormalization(),
            activation,
            kr.layers.Dropout(.2)
        ])
        self.last = kr.Sequential([
            kr.layers.Flatten(),
            kr.layers.Dense(2)
        ])

    def call(self, inputs, training=None, mask=None):
        x = self.embed(inputs)
        x = tf.concat(tf.unstack(x, axis=-2), -1)
        x = self.conv1(x)
        x2 = x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x5 = self.conv5(x2)
        x = self.last(tf.concat([x, x5], -1))
        return x


model = Model()
model_save_path = 'beauty5.ckpt'

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
    if path.exists(f'{model_save_path}.index'):
        model.load_weights(model_save_path)
    cp = kr.callbacks.ModelCheckpoint(model_save_path,
                                      save_weights_only=True,
                                      save_best_only=True,
                                      monitor='val_sparse_categorical_accuracy')
    for beauties, label_0, normals, label_1 in dataset:
        x_train = np.vstack([beauties, normals])
        y_train = np.hstack([label_0, label_1])
        model.fit(x_train, y_train, validation_data=(x_test, y_test), callbacks=cp, workers=12)
