import tensorflow.keras as kr
import tensorflow as tf
from tensorflow.keras.datasets import cifar10
import numpy as np
import matplotlib.pyplot as plt


class GruTransformerTranspose(kr.Model):
    def __init__(self, units, window_size=4, splits=2, activation=None):
        super(GruTransformerTranspose, self).__init__()
        self.block = GruTransformerTransposeBlock(units, window_size, activation)
        self.window_size, self.splits = window_size, splits

    def call(self, inputs, training=None, mask=None):
        if self.splits == 1:
            x = self.block(inputs)
            return x
        _, h, w, c = inputs.shape
        assert h % self.splits == 0 and w % self.splits == 0, 'Error split number!'
        hs, x = tf.split(inputs, self.splits, 1), []
        for h_ in hs:
            ws = tf.split(h_, self.splits, 2)
            for w_ in ws:
                hl, wl = len(hs), len(ws)
                x.append(w_)
        x = tf.stack(x, 1)
        x = tf.reshape(x, [-1, *x.shape[2:]])
        x = self.block(x)
        x = tf.reshape(x, [-1, hl * wl, *x.shape[-3:]])
        x = tf.split(x, hl, 1)
        x = tf.concat(x, 2)
        x = tf.unstack(x, axis=1)
        x = tf.concat(x, 2)
        return x




class GruTransformerTransposeBlock(kr.Model):
    def __init__(self, units, window_size=4, activation=None):
        super(GruTransformerTransposeBlock, self).__init__()
        self.gru = kr.layers.GRU(units, return_sequences=True)
        self.dense = kr.layers.Dense(window_size ** 2, activation=activation)
        self.window_size, self.units = window_size, units

    def call(self, inputs, training=None, mask=None):
        x = tf.unstack(inputs, axis=1)
        n = len(x)
        x = tf.concat(x, 1)
        x = self.gru(x)
        x = tf.expand_dims(x, -1)
        x = self.dense(x)
        x = tf.transpose(x, [0, 1, 3, 2])
        x = tf.reshape(x, [-1, x.shape[1], self.window_size, self.window_size, self.units])
        x = tf.split(x, n, axis=1)
        x = tf.concat(x, 2)
        x = tf.unstack(x, axis=1)
        x = tf.concat(x, 2)
        return x


class GruTransformer(kr.Model):
    def __init__(self, units, window_size=4):
        super(GruTransformer, self).__init__()
        self.block = GruTransformerBlock(units)
        self.window_size, self.units = window_size, units

    def call(self, inputs, training=None, mask=None):
        _, h, w, c = inputs.shape
        nh, nw = h // self.window_size, w // self.window_size
        nh = nh + 1 if nh * self.window_size < h else nh
        nw = nw + 1 if nw * self.window_size < w else nw
        masks = []
        for ih in range(nh):
            if (ih + 1) * self.window_size > h:
                w_mask = inputs[:, -self.window_size:, ...]
            else:
                w_mask = inputs[:, ih * self.window_size:(ih + 1) * self.window_size, ...]
            for iw in range(nw):
                if (iw + 1) * self.window_size > w:
                    mask = w_mask[:, :, -self.window_size:, :]
                else:
                    mask = w_mask[:, :, iw * self.window_size:(iw + 1) * self.window_size, :]
                masks.append(mask)
        masks = tf.stack(masks, 1)
        masks = tf.reshape(masks, [-1, self.window_size, self.window_size, c])
        output = self.block(masks)
        output = tf.reshape(output, [-1, nh, nw, self.units])
        return output


class GruTransformerBlock(kr.Model):
    def __init__(self, units, activation=kr.activations.tanh):
        super(GruTransformerBlock, self).__init__()
        self.gru = kr.layers.GRU(units, activation=activation)

    def call(self, inputs, training=None, mask=None):
        x = tf.unstack(inputs, axis=1)
        x = tf.concat(x, -1)
        x = self.gru(x)
        return x


class StandardScaler:
    def fit_transform(self, data: np.ndarray):
        self.mean = np.mean(data)
        self.std = np.std(data)
        self.dtype = data.dtype
        value = (data - self.mean) / self.std
        return value

    def transform(self, data):
        value = (data - self.mean) / self.std
        return value

    def inverse_transform(self, data):
        value: np.ndarray = data * self.std + self.mean
        return value.astype(self.dtype)


if __name__ == '__main__':
    scaler = StandardScaler()
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    x_train, x_test = scaler.fit_transform(x_train), scaler.transform(x_test)
    model = kr.Sequential([
        GruTransformerTranspose(3, 4),
        GruTransformer(3, 4),
        # kr.layers.Flatten(),
        # kr.layers.Dense(10)
    ])
    xx = model(x_test[-1:])
    plt.subplot(121)
    plt.imshow(xx[0])
    plt.subplot(122)
    plt.imshow(x_test[-1])
    plt.show()
    # model.compile(optimizer=kr.optimizers.Adam(),
    #               loss=kr.losses.SparseCategoricalCrossentropy(True),
    #               metrics=[kr.metrics.sparse_categorical_accuracy])
    # model.fit(x_train, y_train, 64, 5, validation_data=(x_test, y_test))
    model.summary()
    model.layers[0].summary()
