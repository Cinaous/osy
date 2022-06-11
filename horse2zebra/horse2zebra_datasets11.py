import math

import cv2
import numpy as np
import random
import glob
from trm.gru_transformer import StandardScaler


def read_image(image_path, size=(128, 128)):
    image = cv2.imread(image_path)
    image = cv2.resize(image, dsize=size)
    return image


class Datasets:
    def __init__(self, batch=32):
        self.z_scaler, self.h_scaler = StandardScaler(), StandardScaler()
        horses = [read_image(horse) for horse in glob.glob('trainA/*.jpg')]
        self.horses = self.h_scaler.fit_transform(np.array(horses))
        zebras = [read_image(zebra) for zebra in glob.glob('trainB/*.jpg')]
        self.zebras = self.z_scaler.fit_transform(np.array(zebras))
        self.test_horses = glob.glob('testA/*.jpg')
        self.test_zebras = glob.glob('testB/*.jpg')
        self.batch = batch

    def __iter__(self):
        return self

    @staticmethod
    def convert_data(image, scaler: StandardScaler):
        image = scaler.inverse_transform(image)
        return image.astype('uint8')

    def __next__(self):
        self.horses = np.roll(self.horses, -self.batch, 0)
        self.zebras = np.roll(self.zebras, -self.batch, 0)
        return self.horses[:self.batch], self.zebras[:self.batch]

    @staticmethod
    def sample_image(data_paths, batch, scaler: StandardScaler):
        random.seed(None)
        horses = random.sample(data_paths, batch)
        data = []
        for horse in horses:
            horse = read_image(horse)
            horse = scaler.transform(horse)
            data.append(horse)
        return np.array(data)

    def load_test_dataset(self, batch=10):
        return self.sample_image(self.test_horses, batch, self.h_scaler), \
               self.sample_image(self.test_zebras, batch, self.z_scaler)


if __name__ == '__main__':
    datasets = Datasets(1)
    horse, zebra = datasets.load_test_dataset()
    print(horse.shape)
    for horse, zebra in datasets:
        image = np.concatenate((horse, zebra), axis=2)
        cv2.imshow('horse@zebra', image[0])
        cv2.waitKey(10)
