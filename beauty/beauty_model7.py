import tensorflow.keras as kr
from tensorflow.keras.datasets import cifar10
import os.path as path
from pix2pix.pix_models import ConvLayer

model = kr.Sequential([
    ConvLayer(32, 5, 6),
    kr.layers.Flatten(),
    kr.layers.Dense(10)
])
model_save_path = 'beauty7.ckpt'

if __name__ == '__main__':
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    x_train, x_test = x_train / 255., x_test / 255.
    model.compile(optimizer=kr.optimizers.Adam(),
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
    model.fit(x_train, y_train, batch_size=32, epochs=5000,
              validation_data=(x_test, y_test), callbacks=cp,
              workers=12, use_multiprocessing=True)
