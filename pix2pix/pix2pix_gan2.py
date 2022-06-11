import tensorflow.keras as kr
from dataset_pix2pix import Dataset
from pix_models import RestNetModel, RestUNetModel
from pix_scaler import FixedScaler as Scaler
import numpy as np
import tensorflow as tf
import cv2
import os.path as path

GNet = RestUNetModel((18, 3), 96, 3, last_layer=kr.layers.Conv2DTranspose(3, 1))
DNet = RestNetModel((9, 3), 96, 3, last_layer=kr.layers.Conv2D(1, 1))

real_label, fake_label = None, None
optimizer = kr.optimizers.Adam(2e-4)
mse = kr.losses.MeanSquaredError()
dataset = Dataset('../horse2zebra')
scale = Scaler()
file_save_path = './images/gan2'
mds = [GNet, DNet]
models_save_path = [f'./models/{md.name}-2' for md in mds]
for model, save_path in zip(mds, models_save_path):
    if path.exists(save_path + '.index'):
        model.load_weights(save_path)
epoch = 0
for trainA, trainB in dataset:
    if not epoch:
        real_label = np.ones(DNet.compute_output_shape(trainA.shape))
        fake_label = np.zeros_like(real_label)
    if epoch % 25 == 0:
        testA, testB = dataset.load_test_data()
        preB = GNet(scale.fit_transform(testA))
        preB = scale.inverse_transform(preB)
        img = np.hstack([testA[0], preB[0]])
        cv2.imwrite(f'{file_save_path}/gan-{epoch}.jpg', img)
    epoch += 1
    print(f"Current epoch is {epoch}")
    trainA, trainB = scale.fit_transform(trainA), scale.transform(trainB)
    with tf.GradientTape(watch_accessed_variables=False) as tape:
        tape.watch(DNet.variables)
        losses = [mse(real_label, DNet(trainB)), mse(fake_label, DNet(GNet(trainA)))]
        loss = tf.reduce_sum(losses)
    grads = tape.gradient(loss, tape.watched_variables())
    optimizer.apply_gradients(zip(grads, tape.watched_variables()))
    print('Disc loss is %.5f' % loss,
          *[f'output-{i} is %.5f' % loss for i, loss in enumerate(losses)], sep=', ', end='[ ')
    with tf.GradientTape(watch_accessed_variables=False) as tape:
        tape.watch(GNet.variables)
        loss = mse(real_label, DNet(GNet(trainA)))
    grads = tape.gradient(loss, tape.watched_variables())
    optimizer.apply_gradients(zip(grads, tape.watched_variables()))
    print('Gen loss is %.5f' % loss, end=']\r\n')
    try:
        [model.save_weights(save_path) for model, save_path in zip(mds, models_save_path)]
    except Exception as ex:
        print(ex)
