import os.path as path
from abc import ABC
import tensorflow as tf
import tensorflow.keras as kr
from tensorflow.keras.datasets import mnist


class Sample(kr.Model, ABC):

    def __init__(self, n_features, activation=tf.nn.leaky_relu):
        super(Sample, self).__init__()
        self.Q = kr.layers.Dense(n_features, activation)
        self.K = kr.layers.Dense(n_features, activation)
        self.V = kr.layers.Dense(n_features, activation)

    def call(self, inputs, training=None, mask=None):
        q = self.Q(inputs)
        k = self.K(inputs)
        v = self.V(inputs)
        kt = tf.transpose(k, perm=[0, 2, 1])
        s = tf.matmul(q, kt)
        s = kr.activations.softmax(s)
        out = tf.matmul(s, v)
        return out


class Transformer(kr.Model, ABC):
    def __init__(self, n_features, n_outputs, n_headers=8):
        super(Transformer, self).__init__()
        self.hs = [Sample(n_features) for _ in range(n_headers)]
        self.d = kr.Sequential([
            kr.layers.Dense(n_outputs),
            kr.layers.LayerNormalization(),
            kr.layers.LeakyReLU()
        ])

    def call(self, inputs, training=None, mask=None):
        hs = [h(inputs) for h in self.hs]
        out = tf.concat(hs, -1)
        out = self.d(out)
        return out


class ImageModel(kr.Model, ABC):
    def __init__(self):
        super(ImageModel, self).__init__()
        self.trm1 = Transformer(64, 128)
        self.trm2 = Transformer(32, 64)
        self.trm3 = Transformer(16, 32)
        self.trm4 = Transformer(8, 16)
        self.f = kr.layers.Flatten()
        self.d = kr.layers.Dense(10, activation=kr.activations.softmax)

    def call(self, inputs, training=None, mask=None):
        out = self.trm1(inputs)
        out = self.trm2(out)
        out = self.trm3(out)
        out = self.trm4(out)
        out = self.f(out)
        out = self.d(out)
        return out


if __name__ == '__main__':
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train, x_test = x_train / 127.5 - 1., x_test / 127.5 - 1.
    model = ImageModel()
    model.compile(optimizer=kr.optimizers.Adam(),
                  loss=kr.losses.SparseCategoricalCrossentropy(),
                  metrics=[kr.metrics.sparse_categorical_accuracy])
    model_path = 'model.ckpt'
    model_cp = kr.callbacks.ModelCheckpoint(model_path, save_best_only=True, save_weights_only=True,
                                            monitor='sparse_categorical_accuracy')
    if path.exists(model_path + '.index'):
        model.load_weights(model_path)
    model.fit(x_train, y_train, 64, epochs=20000, validation_data=(x_test, y_test), validation_freq=1,
              callbacks=[model_cp], validation_batch_size=128)
    model.summary()
