import tensorflow.keras as kr
from tensorflow.keras.datasets import cifar10
import os.path as path
from pix2pix.pix_scaler import M3Scaler as Scaler
from pix2pix.pix_models import ConvLayer

model = kr.Sequential([
    ConvLayer(32, 5, 5, strides=2, padding='same'),
    kr.layers.Flatten(),
    kr.layers.Dense(10)
])
model_save_path = 'beauty10.ckpt'
scale = Scaler()

if __name__ == '__main__':
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    x_train, x_test = scale.fit_transform(x_train), scale.fit_transform(x_test)
    model.compile(optimizer=kr.optimizers.Adam(),
                  loss=kr.losses.SparseCategoricalCrossentropy(True),
                  metrics=[kr.metrics.sparse_categorical_accuracy])
    model.build([None, *x_test.shape[1:]])
    model.summary()
    if path.exists(f'{model_save_path}.index'):
        model.load_weights(model_save_path)
    cp = kr.callbacks.ModelCheckpoint(model_save_path, save_weights_only=True)
    model.fit(x_train, y_train, batch_size=32, epochs=5000,
              validation_data=(x_test, y_test), callbacks=cp,
              workers=12, use_multiprocessing=True)
