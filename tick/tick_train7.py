import abc
import numpy as np
from tick_bussing2 import Dataset
import tensorflow.keras as kr
import tensorflow as tf
import os.path as path
import matplotlib.pyplot as plt


class RN3Net(kr.Model, abc.ABC):
    def __init__(self, filters=64, unit=kr.layers.GRU,
                 deep=17, activation=kr.activations.tanh, dropout=.2, last_layer=None):
        super(RN3Net, self).__init__()
        self.units = [(unit(filters, return_sequences=i != deep - 1),
                       kr.layers.Dense(filters)) for i in range(deep)]
        self.norm = [kr.layers.LayerNormalization() for _ in range(deep)]
        self.activation = activation
        self.dropout = kr.layers.Dropout(dropout)
        self.last_layer = last_layer

    def build(self, input_shape):
        self.last_layer = self.last_layer or kr.layers.Dense(input_shape[-1])

    def call(self, inputs, training=None, mask=None):
        cx = rx = x = inputs
        for i, (unit, liner) in enumerate(self.units):
            if i % 2 == 0 and i:
                x += rx
                rx = cx
            cx = x = liner(unit(self.dropout(self.activation(self.norm[i - 1](x))) if i else x))
            rx = rx if i else x
        return self.last_layer(self.activation(self.norm[-1](x)))


class SeqNet(kr.Model, abc.ABC):
    def __init__(self, output_dim, rnn, seq=5):
        super(SeqNet, self).__init__()
        self.rnn = rnn
        self.seq = seq
        self.last_layer = kr.layers.Dense(output_dim)

    def call(self, inputs, training=None, mask=None):
        x, xs = inputs, tf.unstack(inputs, axis=1)
        xx = [self.rnn(x)]
        for i in range(1, self.seq):
            x = tf.stack([*xs[i:], *xx], 1)
            xx.append(self.rnn(x))
        x = tf.stack(xx, 1)
        x = self.last_layer(x)
        return x


Net = SeqNet(1, RN3Net())
dataset = Dataset('000001.SZ', seq_num=30, batch=128)
net_save_path = 'mdl/seq7.net'
if path.exists(f'{net_save_path}.index'):
    print('------------------------- loading model -------------------------')
    Net.load_weights(net_save_path)
if __name__ == '__main__':
    print('train start....')
    cp = kr.callbacks.ModelCheckpoint(net_save_path, save_weights_only=True, save_best_only=True)
    Net.compile(optimizer=kr.optimizers.Adamax(), loss=kr.losses.mse)
    epoch = 0
    for x_train, y_train in dataset:
        if epoch % 25 == 0:
            x_test, y_test = dataset.load_test_data()
            y_pred = Net.predict(x_test)
            x_test = x_test[0, -1:, 3:4]
            y_pred = np.concatenate((x_test, y_pred[0]), 0)
            y_test = np.concatenate((x_test, y_test[0]), 0)
            plt.plot(y_pred)
            plt.plot(y_test)
            plt.legend(['predict', 'real'])
            plt.savefig(f'fig7/{epoch}-train.jpg')
            plt.close()
        epoch += 1
        print(f'current epoch is {epoch}')
        x_test, y_test = dataset.load_test_data(64)
        Net.fit(x_train, y_train, 16, workers=12, use_multiprocessing=True, callbacks=cp,
                validation_data=(x_test, y_test))
