import tensorflow.keras as kr
import tensorflow as tf
from tensorflow.keras.datasets import cifar10
from horse2zebra.horse2zebra_datasets12 import StandardScaler
import numpy as np

if __name__ == '__main__':
    scaler = StandardScaler()
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    x_train, x_test = scaler.fit_transform(x_train), scaler.fit_transform(x_test)

    out = np.einsum('...hwc,...hwc->...wh', x_train[-1:], x_train[-1:])
    print(out)
