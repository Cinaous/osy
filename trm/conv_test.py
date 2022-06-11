from conv_Models import samples
from tensorflow.keras.datasets import mnist
import numpy as np
import matplotlib.pyplot as plt

if __name__ == '__main__':
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train, x_test = x_train[..., np.newaxis] / 127.5 - 1., x_test[..., np.newaxis] / 127.5 - 1.
    model = samples([(32, 5, 2, None),
                     (128, 3, 2, .2),
                     (512, 3, 2, .5)], 10)
    model_save_path = 'samples.ckpt'
    model.load_weights(model_save_path)
    x_test = x_test[:12]
    p = model.predict(x_test)
    for i in range(12):
        plt.subplot(3, 4, i + 1)
        plt.imshow(x_test[i])
        plt.title('p%d' % np.argmax(p[i]))
        plt.xticks([])
        plt.yticks([])
    plt.show()
