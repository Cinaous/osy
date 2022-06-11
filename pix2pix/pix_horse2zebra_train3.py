from tensorflow_examples.models.pix2pix import pix2pix
import tensorflow as tf
import cv2
import numpy as np
from pix2pix.dataset_horse2zebra import data_save_dir, label_trainA, label_trainB
from pix_scaler import M3Scaler as Scaler

if __name__ == '__main__':
    epochs = 1
    batch_size = 1
    enable_function = True
    scale = Scaler()

    input_image = np.load(f'{data_save_dir}/{label_trainA}.npy')
    target_image = np.load(f'{data_save_dir}/{label_trainB}.npy')
    input_image = scale.fit_transform(input_image)[0]
    target_image = scale.fit_transform(target_image)[0]

    train_dataset = tf.data.Dataset.from_tensors(
        (input_image, target_image)).map(pix2pix.random_jitter).batch(
        batch_size)
    checkpoint_pr = pix2pix.get_checkpoint_prefix()

    pix2pix_obj = pix2pix.Pix2pix(epochs, enable_function)
    pix2pix_obj.train(train_dataset, checkpoint_pr)
    gen_image = pix2pix_obj.generator(input_image)
    gen_image = scale.inverse_transform(gen_image)
    cv2.imshow('gen', gen_image[0])
    cv2.waitKey(10)
