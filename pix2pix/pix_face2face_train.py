import cv2
import numpy as np
from dataset_face2face import Dataset
from pix_models import Pix2PixUNet, Pix2PixEncoder
import tensorflow.keras as kr
import os.path as path
from pix_scaler import M3Scaler
import tensorflow as tf

A2B = Pix2PixUNet()
B2A = Pix2PixUNet()
DA = Pix2PixEncoder()
DB = Pix2PixEncoder()
for model in [A2B, B2A, DA, DB]:
    model.build([None, 8, 8, 3])
    model.summary()
lr = 2e-4
discriminator_optimizer = kr.optimizers.Adam(lr)
cycle_optimizer = kr.optimizers.Adam(lr)
mae, mse = kr.losses.MeanAbsoluteError(), kr.losses.MeanSquaredError()
a2b_save_path = './models/face-a2b.ckpt'
b2a_save_path = './models/face-b2a.ckpt'
da_save_path = './models/face-da.ckpt'
db_save_path = './models/face-db.ckpt'
# if path.exists(a2b_save_path + '.index'):
#     A2B.load_weights(a2b_save_path)
# if path.exists(b2a_save_path + '.index'):
#     B2A.load_weights(b2a_save_path)
# if path.exists(da_save_path + '.index'):
#     DA.load_weights(da_save_path)
# if path.exists(db_save_path + '.index'):
#     DB.load_weights(db_save_path)

dataset = Dataset()
scaler = M3Scaler()
epoch = 0
train_data_save_path = './images/face2face'
for trainA, trainB in dataset:
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
    print(f'Current epoch is {epoch}:')
    trainA, trainB = scaler.fit_transform(trainA), scaler.fit_transform(trainB)
    real_label = np.ones(DA.compute_output_shape(trainA.shape))
    fake_label = np.zeros_like(real_label)
    with tf.GradientTape(watch_accessed_variables=False) as tape:
        tape.watch(DA.variables + DB.variables)
        sub_loss = [mse(y_true, y_pred) for y_true, y_pred in
                    zip([real_label, real_label, fake_label, fake_label],
                        [DA(trainA), DB(trainB), DA(B2A(trainB)), DB(A2B(trainA))])]
        loss = tf.reduce_sum(sub_loss)
    print('\tDiscriminator loss is %.5f;' % loss, end=' ')
    [print('Outputs-%d is %.5f; ' % (i, sl), end=' ' if i != 3 else '\r\n') for i, sl in enumerate(sub_loss)]
    grads = tape.gradient(loss, tape.watched_variables())
    discriminator_optimizer.apply_gradients(zip(grads, tape.watched_variables()))
    with tf.GradientTape(watch_accessed_variables=False) as tape:
        tape.watch(A2B.variables + B2A.variables)
        a2b = A2B(trainA)
        b2a = B2A(trainB)
        sub_loss = [loss_fn(y_true, y_pred) for loss_fn, y_true, y_pred in
                    zip([mae, mae, mse, mse, mae, mae],
                        [trainA, trainB, real_label, real_label, trainA, trainB],
                        [B2A(a2b), A2B(b2a), DA(b2a), DB(a2b), B2A(trainA), A2B(trainB)])]
        loss = tf.reduce_sum(tf.multiply(sub_loss, [10, 10, 1, 1, 5, 5]))
    print('\tCycle loss is %.5f;' % loss, end=' ')
    [print('Outputs-%d is %.5f; ' % (i, sl), end=' ' if i != 5 else '\r\n') for i, sl in enumerate(sub_loss)]
    grads = tape.gradient(loss, tape.watched_variables())
    cycle_optimizer.apply_gradients(zip(grads, tape.watched_variables()))
    try:
        A2B.save_weights(a2b_save_path)
        B2A.save_weights(b2a_save_path)
        DA.save_weights(da_save_path)
        DB.save_weights(db_save_path)
    except Exception as ex:
        print(ex)
