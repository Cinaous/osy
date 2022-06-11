import cv2
import numpy as np
from dataset_pix2pix import Dataset
from pix_models import RestNetModel, RestUNetModel
import tensorflow.keras as kr
import os.path as path
from pix_scaler import FixedScaler as Scaler
import tensorflow as tf

A2B = RestUNetModel((18, 3), 128, 3, last_layer=kr.layers.Conv2DTranspose(3, 1))
B2A = RestUNetModel((18, 3), 128, 3, last_layer=kr.layers.Conv2DTranspose(3, 1))
DA = RestNetModel((9, 3), 128, 3, last_layer=kr.layers.Conv2D(1, 1))
DB = RestNetModel((9, 3), 128, 3, last_layer=kr.layers.Conv2D(1, 1))

lr, label, lambdas = 2e-4, 10, [5, 5, 1, 1, 2.5, 2.5]
d_optimizer, c_optimizer = kr.optimizers.Adam(lr), kr.optimizers.Adam(lr)
mae, mse = kr.losses.MeanAbsoluteError(), kr.losses.MeanSquaredError()
models_save_path, models = [f'./models/cycle-{label}-{i + 1}' for i in range(4)], [A2B, B2A, DA, DB]
for model, model_save_path in zip(models, models_save_path):
    if path.exists(model_save_path + '.index'):
        model.load_weights(model_save_path)

dataset = Dataset('../horse2zebra')
real_label, fake_label, epoch, scaler = None, None, 0, Scaler()
train_data_save_path = f'./images/horse2zebra{label}'
for trainA, trainB in dataset:
    if epoch == 0:
        real_label = np.ones(DA.compute_output_shape(trainA.shape))
        fake_label = np.zeros_like(real_label)
        B2A.build(trainB.shape)
        DA.summary(), B2A.summary()
    if epoch % 25 == 0:
        testA, testB = dataset.load_test_data()
        test_a2b = A2B(scaler.fit_transform(testA))
        test_a2b2a = B2A(test_a2b)
        test_a2b, test_a2b2a = scaler.inverse_transform(test_a2b), scaler.inverse_transform(test_a2b2a)
        image_A = np.hstack([testA[0], test_a2b[0], test_a2b2a[0]])
        test_b2a = B2A(scaler.fit_transform(testB))
        test_b2a2b = A2B(test_b2a)
        test_b2a, test_b2a2b = scaler.inverse_transform(test_b2a), scaler.inverse_transform(test_b2a2b)
        image_B = np.hstack([testB[0], test_b2a[0], test_b2a2b[0]])
        image_AB = np.vstack([image_A, image_B])
        cv2.imwrite(f'{train_data_save_path}/{epoch}-train.jpg', image_AB)
    epoch += 1
    print(f'current epoch is {epoch}: {trainA.shape}')
    trainA, trainB = scaler.fit_transform(trainA), scaler.fit_transform(trainB)
    with tf.GradientTape(watch_accessed_variables=False) as tape:
        tape.watch(A2B.variables + B2A.variables)
        a2b = A2B(trainA)
        b2a = B2A(trainB)
        losses = [mae(trainA, B2A(a2b)), mae(trainB, A2B(b2a)), mse(real_label, DA(b2a)),
                  mse(real_label, DB(a2b)), mae(trainA, B2A(trainA)), mae(trainB, A2B(trainB))]
        loss = tf.reduce_sum(tf.multiply(losses, lambdas))
    assert not np.isnan(loss)
    grads = tape.gradient(loss, tape.watched_variables())
    c_optimizer.apply_gradients(zip(grads, tape.watched_variables()))
    print('Cycle loss is %.5f' % loss, *[f', output-{i + 1} is %.5f' % loss for i, loss in enumerate(losses)], sep='')
    with tf.GradientTape(watch_accessed_variables=False) as tape:
        tape.watch(DA.variables + DB.variables)
        losses = [mse(real_label, DA(trainA)), mse(real_label, DB(trainB)),
                  mse(fake_label, DA(b2a)), mse(fake_label, DB(a2b))]
        loss = tf.reduce_sum(losses)
    assert not np.isnan(loss)
    grads = tape.gradient(loss, tape.watched_variables())
    d_optimizer.apply_gradients(zip(grads, tape.watched_variables()))
    print('Dis loss is %.5f' % loss, *[f', output-{i + 1} is %.5f' % loss for i, loss in enumerate(losses)], sep='')
    try:
        [model.save_weights(model_save_path) for model, model_save_path in zip(models, models_save_path)]
    except Exception as ex:
        print(ex)
