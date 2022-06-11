from tensorflow.keras.datasets import cifar10
from conv_Models2 import samples
import matplotlib.pyplot as plt
import numpy as np

if __name__ == '__main__':
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    x_train, x_test = x_train / 127.5 - 1., x_test / 127.5 - 1.
    model = samples([(64, 5, 2, None),
                     (128, 3, 2, .2),
                     (512, 3, 2, .5)], [256, 128, 10])
    model_save_path = 'samples2.ckpt'
    model.load_weights(model_save_path)
    labels = ['airplane',
              'automobile',
              'bird',
              'cat',
              'deer',
              'dog',
              'frog',
              'horse',
              'ship',
              'truck']

    while True:
        ixs = np.random.choice(len(x_test), 12)
        x_p = x_test[ixs]
        p = model.predict(x_p)
        for i in range(12):
            plt.subplot(3, 4, i + 1)
            plt.imshow(x_p[i])
            plt.xticks([])
            plt.yticks([])
            plt.title(labels[np.argmax(p[i])])
        plt.show()
