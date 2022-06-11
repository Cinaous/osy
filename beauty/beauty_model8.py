import tensorflow.keras as kr
from preprocessing_dataset3 import Dataset
import numpy as np
import os.path as path
from transformer.swin_s import Transformer

model = kr.Sequential([
    Transformer(4, 96, 8, 5),
    kr.layers.Flatten(),
    kr.layers.Dense(2)
])
model_save_path = 'beauty8.ckpt'

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
        model.fit(x_train, y_train, validation_data=(x_test, y_test),
                  callbacks=cp, workers=12, use_multiprocessing=True)
