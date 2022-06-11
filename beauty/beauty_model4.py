import tensorflow.keras as kr
from preprocessing_dataset2 import Dataset
import numpy as np
import os.path as path

activation = kr.layers.Activation(kr.activations.tanh)


def conv2d(units, num):
    return kr.Sequential([
        *[kr.layers.Conv2D(units, 3, padding='same') for _ in range(num)],
        kr.layers.LayerNormalization(),
        activation,
        kr.layers.MaxPool2D(),
        kr.layers.Dropout(.2)
    ])


model = kr.Sequential([
    # 4 * 48
    *[conv2d(48, i + 1) for i in range(5)],
    # 2
    kr.layers.Flatten(),
    kr.layers.Dense(2)
])
model_save_path = 'beauty4.ckpt'

if __name__ == '__main__':
    dataset = Dataset()
    beauties, label_0, normals, label_1 = dataset.load_test()
    x_test = np.vstack([beauties, normals])
    y_test = np.hstack([label_0, label_1])
    model.compile(optimizer=kr.optimizers.Adam(),
                  loss=kr.losses.SparseCategoricalCrossentropy(True),
                  metrics=[kr.metrics.sparse_categorical_accuracy])
    model.build([None, *x_test.shape[1:]])
    model.summary()
    # if path.exists(f'{model_save_path}.index'):
    #     model.load_weights(model_save_path)
    cp = kr.callbacks.ModelCheckpoint(model_save_path,
                                      save_weights_only=True,
                                      save_best_only=True)
    for beauties, label_0, normals, label_1 in dataset:
        x_train = np.vstack([beauties, normals])
        y_train = np.hstack([label_0, label_1])
        model.fit(x_train, y_train, validation_data=(x_test, y_test), callbacks=cp)
