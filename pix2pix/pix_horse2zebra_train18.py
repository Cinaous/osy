import cv2
import numpy as np
from dataset3_pix2pix import Dataset
from pix_models import RestUNetPLayer, RestNetPLayer, CycleNet, DiscriminatorNet
import tensorflow.keras as kr
import os.path as path
from pix_scaler import FsScaler as Scaler

A2B = RestUNetPLayer((15, 3), 48, 3, last_layer=kr.layers.Conv2DTranspose(3, 3, padding='same'))
B2A = RestUNetPLayer((15, 3), 48, 3, last_layer=kr.layers.Conv2DTranspose(3, 3, padding='same'))
DA = RestNetPLayer((8, 3), 48, 3, last_layer=kr.layers.Conv2D(1, 3, padding='same'))
DB = RestNetPLayer((8, 3), 48, 3, last_layer=kr.layers.Conv2D(1, 3, padding='same'))

lr = 2e-4
cycleModel = CycleNet(A2B, B2A, DA, DB)
discriminatorModel = DiscriminatorNet(A2B, B2A, DA, DB)
cycle_save_path = './models/cycle18.ckpt'
discriminator_save_path = './models/discriminator18.ckpt'
if path.exists(cycle_save_path + '.index'):
    cycleModel.load_weights(cycle_save_path)
if path.exists(discriminator_save_path + '.index'):
    discriminatorModel.load_weights(discriminator_save_path)

dataset = Dataset('../horse2zebra/faces')
scaler = Scaler()
real_label = None
fake_label = None

epoch = 0
train_data_save_path = './images/horse2zebra18'
for trainA, trainB in dataset:
    real_label = np.ones(DA.compute_output_shape(trainA.shape)) if real_label is None else real_label
    fake_label = np.zeros_like(real_label) if fake_label is None else fake_label
    if epoch % 25 == 0:
        testA, testB = trainA, trainB
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
    print(f'current epoch is {epoch}')
    trainA, trainB = scaler.fit_transform(trainA), scaler.fit_transform(trainB)
    discriminatorModel.fit([trainA, trainB], [real_label, real_label, fake_label, fake_label])
    cycleModel.fit([trainA, trainB], [trainA, trainB, real_label, real_label])
    try:
        discriminatorModel.save_weights(discriminator_save_path)
        cycleModel.save_weights(cycle_save_path)
    except Exception as ex:
        print(ex)
