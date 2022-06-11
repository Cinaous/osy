import matplotlib.pyplot as plt
import numpy as np
from tick_train3 import Net, dataset
from datetime import datetime

if __name__ == '__main__':
    x, (x_mean, x_std) = dataset.load_predict_data()
    y = Net.predict(x)
    x = x[0, :, 3:4]
    y = np.concatenate([x, y[0]], 0)
    y = y * x_std[0] + x_mean[0]
    plt.subplot(211)
    plt.plot(y)
    plt.plot(y[:-5])
    plt.legend(['X', 'Y'])
    plt.subplot(212)
    plt.plot(y[-6:])
    plt.legend(['Z'])
    plt.savefig(f'predict/3-{datetime.now().strftime("%Y-%m-%d.%H.%M.%S")}.jpg')
    plt.show()
    plt.close()
