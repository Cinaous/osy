import tensorflow.keras as kr
from dataset_pix2pix import Dataset
from pix_models import RestNetModel, RestUNetModel
from pix_scaler import StandardScaler as Scaler
import numpy as np
import tensorflow as tf
import cv2
import os.path as path
import time

A2B = RestUNetModel((9, 3), 128, 3, last_layer=kr.layers.Conv2DTranspose(3, 1))
B2A = RestUNetModel((9, 3), 128, 3, last_layer=kr.layers.Conv2DTranspose(3, 1))
DA = RestNetModel((9, 3), 128, 3, last_layer=kr.layers.Conv2D(1, 1))
DB = RestNetModel((9, 3), 128, 3, last_layer=kr.layers.Conv2D(1, 1))

real_label, fake_label = None, None
optimizer = kr.optimizers.Adam(2e-4)
mse, mae = kr.losses.MeanSquaredError(), kr.losses.MeanAbsoluteError()
dataset = Dataset('../horse2zebra', 128)
scale = Scaler()
file_save_path = './images/gan3'
mds = [A2B, B2A, DA, DB]
models_save_path = [f'./models/{md.name}.3{i}' for i, md in enumerate(mds)]
for model, save_path in zip(mds, models_save_path):
    if path.exists(save_path + '.index'):
        model.load_weights(save_path)
epoch = 0
for trainA, trainB in dataset:
    if not epoch:
        real_label = np.ones(DA.compute_output_shape(trainA.shape))
        fake_label = np.zeros_like(real_label)
    if epoch % 25 == 0:
        testA, testB = dataset.load_test_data()
        a2b = A2B(scale.fit_transform(testA))
        b2a = B2A(a2b)
        img1 = np.hstack([testA[0], scale.inverse_transform(a2b)[0], scale.inverse_transform(b2a)[0]])
        a2b = B2A(scale.fit_transform(testB))
        b2a = A2B(a2b)
        img2 = np.hstack([testB[0], scale.inverse_transform(a2b)[0], scale.inverse_transform(b2a)[0]])
        img = np.vstack([img1, img2])
        cv2.imwrite(f'{file_save_path}/gan-{epoch}.jpg', img)
    epoch += 1
    print(f"Current epoch is {epoch}:", end=" ")
    trainA, trainB = scale.fit_transform(trainA), scale.transform(trainB)
    start = time.time()
    with tf.GradientTape(True) as tape:
        a2b, b2a = A2B(trainA), B2A(trainB)
        d_losses = [mse(real_label, DA(trainA)), mse(real_label, DA(trainB)),
                    mse(fake_label, DA(b2a)), mse(fake_label, DB(a2b))]
        g_losses = [10. * mae(trainA, B2A(a2b)), 10. * mae(trainB, A2B(b2a)),
                    mse(real_label, DA(b2a)), mse(real_label, DB(a2b))]
        model_loss = (tf.reduce_sum(d_losses), tf.reduce_sum(g_losses))
    [optimizer.apply_gradients(zip(tape.gradient(ml, v), v)) for ml, v in
     zip(model_loss, (DA.trainable_variables + DB.trainable_variables,
                      A2B.trainable_variables + B2A.trainable_variables))]
    print('<%.3fs>' % (time.time() - start), 'Disc loss is %.5f, Cycle loss is %.5f\n\tDis ->' % model_loss,
          *[f'output-{i} is %.5f' % loss for i, loss in enumerate(d_losses)], "\n\tCycle ->",
          *[f'output-{i} is %.5f' % loss for i, loss in enumerate(g_losses)])
    try:
        [model.save_weights(save_path) for model, save_path in zip(mds, models_save_path)]
    except Exception as ex:
        print(ex)
