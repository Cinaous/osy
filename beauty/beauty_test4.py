from beauty.beauty_model13 import model, model_save_path, scale
from tensorflow.keras.datasets import cifar10
import numpy as np
import matplotlib.pyplot as plt
import tensorflow.keras as kr

model.load_weights(model_save_path)

_, (x_test, y_test) = cifar10.load_data()
ids = np.random.choice(np.arange(len(x_test)), 12, replace=False)
y_test = y_test[ids]
x_test = x_test[ids]
y_pred = model(scale.fit_transform(x_test))
acc = kr.metrics.sparse_categorical_accuracy(y_test, y_pred)
print(acc)
y_pred = np.argmax(y_pred, -1)
for i in range(12):
    plt.subplot(3, 4, i + 1)
    plt.imshow(x_test[i])
    plt.title(f'P{y_pred[i]}=>R{y_test[i]}')
    plt.xticks([])
    plt.yticks([])
plt.show()

