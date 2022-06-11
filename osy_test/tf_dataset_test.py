from tensorflow.keras.preprocessing.image import ImageDataGenerator
import cv2
import numpy as np
from tensorflow.keras.datasets import cifar10

(x_train, y_train), (x_test, y_test) = cifar10.load_data()

train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    rotation_range=360,
    vertical_flip=True)
for x, y in zip(train_datagen.flow(x_train, batch_size=32), train_datagen.flow(x_test, batch_size=32)):
    cv2.imshow('x', cv2.resize(np.hstack((x[0], y[0])), (0, 0), fx=4, fy=4))
    cv2.waitKey(10)
