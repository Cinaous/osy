import abc
import numpy as np
from tick_bussing2 import Dataset
import tensorflow.keras as kr
import os.path as path
import matplotlib.pyplot as plt


class Transformer(kr.Model, abc.ABC):
    def __init__(self):
        super(Transformer, self).__init__()
        self.attention = [(
            kr.layers.MultiHeadAttention(8, 32, dropout=.2),
            kr.layers.LayerNormalization(),
            kr.layers.Dense(32)
        ) for _ in range(5)]
        self.activation = kr.activations.swish
        self.last = kr.layers.MultiHeadAttention(8, 32)
        self.last_layer = kr.layers.Dense(1)

    def call(self, inputs, training=None, mask=None):
        x = inputs
        for attention, norm, liner in self.attention:
            x = attention(x[:, 5:], x)
            x = liner(self.activation(norm(x)))
        x = self.last(x, x)
        x = self.last_layer(x)
        return x


Net = Transformer()
net_save_path = 'mdl/seq5.net'
if path.exists(f'{net_save_path}.index'):
    print('------------------------- loading model -------------------------')
    Net.load_weights(net_save_path)
dataset = Dataset('000001.SZ', seq_num=30, batch=128)
if __name__ == '__main__':
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
            plt.savefig(f'fig5/{epoch}-train.jpg')
            plt.close()
        epoch += 1
        print(f'current epoch is {epoch}')
        x_test, y_test = dataset.load_test_data(64)
        Net.fit(x_train, y_train, 32, workers=12, use_multiprocessing=True, callbacks=cp,
                validation_data=(x_test, y_test))
