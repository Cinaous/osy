from abc import ABC
import tensorflow.keras as kr
from tensorflow.keras.datasets import cifar10
import os.path as path
import tensorflow as tf


class MultiConv(kr.Model, ABC):
    def __init__(self):
        super(MultiConv, self).__init__()
        self.conv = [SplitDimsLayer(1 + i, 96, 5, 2) for i in range(4)]

    def call(self, inputs, training=None, mask=None):
        x = inputs
        for conv in self.conv:
            x = conv(x)
        return x


def split_dims(c, n):
    nums = []
    q, w = c // n, c % n
    for i in range(n):
        if i != n - 1:
            nums.append(q)
        else:
            nums.append(q + w)
    return nums


class SplitDimsLayer(kr.layers.Layer):
    def __init__(self, n, units, kernel_size, strides, padding='same',
                 normalize=kr.layers.LayerNormalization,
                 activation=kr.layers.PReLU, dropout=.2,
                 conv=kr.layers.Conv2D):
        super(SplitDimsLayer, self).__init__()
        self.n = n
        self.conv = [conv(units, kernel_size, strides, padding) for _ in range(n)]
        self.norm = None if normalize is None else normalize()
        self.activation = None if activation is None else activation()
        self.drop = None if dropout is None else kr.layers.Dropout(dropout)
        self.nums = None

    def call(self, inputs, *args, **kwargs):
        c = inputs.shape[-1]
        self.nums = self.nums or split_dims(c, self.n)
        x = tf.split(inputs, self.nums, axis=-1)
        x = [conv(x) for conv, x in zip(self.conv, x)]
        x = tf.concat(x, -1)
        if self.activation is None:
            return x
        x = self.norm(x) if self.norm is not None else x
        x = self.activation(x)
        x = self.drop(x) if self.drop is not None else x
        return x


model = kr.Sequential([
    MultiConv(),
    kr.layers.Flatten(),
    kr.layers.Dense(10)
])
model_save_path = 'beauty9.ckpt'

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
    cp = kr.callbacks.ModelCheckpoint(model_save_path, save_weights_only=True)
    model.fit(x_train, y_train, batch_size=32, epochs=10000,
              validation_data=(x_test, y_test), callbacks=cp,
              workers=12, use_multiprocessing=True)
