import abc

import tensorflow.keras as kr
from tensorflow.keras.datasets import cifar10
import os.path as path
from pix2pix.pix_scaler import FsScaler as Scaler
from pix2pix.pix_models import RestNetModel, CombinationNet, RestNetPLayer, ComplexNet, SelfRestNet


class Net(kr.Model, abc.ABC):
    def __init__(self):
        super(Net, self).__init__()
        self.combination = CombinationNet([
            RestNetModel((6, 3), 32, 3),
            RestNetPLayer((6, 3), 32, 3)
        ], 32)
        self.complex = ComplexNet((3, 2),
                                  lambda: SelfRestNet(3, 32, 3),
                                  last_layer=CombinationNet([
                                      RestNetPLayer((4, 2), 32, 3),
                                      RestNetModel((6, 3), 32, 3)
                                  ], 32))
        self.liner = kr.Sequential([
            kr.layers.Flatten(),
            kr.layers.Dense(10)
        ])

    def call(self, inputs, training=None, mask=None):
        x = self.combination(inputs)
        x = self.complex(x)
        x = self.liner(self.complex.ab(x))
        return x


model = Net()
model_save_path = 'beauty16.ckpt'
scale = Scaler()

if __name__ == '__main__':
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    x_train, x_test = scale.fit_transform(x_train), scale.fit_transform(x_test)
    model.compile(optimizer=kr.optimizers.Adamax(2e-4),
                  loss=kr.losses.SparseCategoricalCrossentropy(True),
                  metrics=[kr.metrics.sparse_categorical_accuracy])
    model.build([None, *x_test.shape[1:]])
    model.summary()
    if path.exists(f'{model_save_path}.index'):
        model.load_weights(model_save_path)
    cp = kr.callbacks.ModelCheckpoint(model_save_path, save_weights_only=True)
    model.fit(x_train, y_train, batch_size=16, epochs=5000,
              validation_data=(x_test, y_test), callbacks=cp,
              workers=12, use_multiprocessing=True)
