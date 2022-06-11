import numpy as np
import tensorflow.keras as kr
from tensorflow.keras.datasets import cifar10
import os.path as path
from pix2pix.pix_scaler import StandardScaler as Scaler
from pix2pix.pix_models import RestNetModel
import tensorflow as tf
import time

model = RestNetModel((7, 3), 48, 3, last_layer=kr.Sequential([
    kr.layers.Flatten(),
    kr.layers.Dense(10)
]))
model_save_path = 'beauty14.ckpt'
scale = Scaler()

if __name__ == '__main__':
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    x_train, x_test = scale.fit_transform(x_train), scale.fit_transform(x_test)
    optimizer, loss_fn = kr.optimizers.Adam(), kr.losses.SparseCategoricalCrossentropy(True)
    metrics_fn = kr.metrics.SparseCategoricalAccuracy()
    model.build([None, *x_test.shape[1:]])
    model.summary()
    if path.exists(f'{model_save_path}.index'):
        model.load_weights(model_save_path)
    epoch = 0
    while True:
        idx = np.random.choice(len(x_train), 32)
        start = time.time()
        with tf.GradientTape() as tape:
            loss = loss_fn(y_train[idx], model(x_train[idx]))
        grads = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))
        model.save_weights(model_save_path)
        idx = np.random.choice(len(x_test), 32)
        y_true = y_test[idx]
        y_pred = model(x_test[idx])
        acc = metrics_fn(y_true, y_pred)
        t_loss = loss_fn(y_true, y_pred)
        print(f'Current epoch is {epoch}: <%.3fs> loss is %.5f, t_loss is %.5f, acc is %.5f'
              % (time.time() - start, loss, t_loss, acc))
        epoch += 1
