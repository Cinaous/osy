import math
import os

import cv2
import numpy as np
import random
import glob
import os.path as path


class StandardScaler:
    def fit_transform(self, data):
        data = np.array(data)
        axis = [i for i in range(len(data.shape) - 1)]
        axis = tuple(axis)
        self.mean = np.mean(data, axis=axis)
        self.std = np.std(data, axis=axis)
        self.dtype = data.dtype
        value = (data - self.mean) / self.std
        return value

    def transform(self, data):
        value = (data - self.mean) / self.std
        return value

    def inverse_transform(self, data):
        data = data.numpy() if hasattr(data, 'numpy') else data
        value = data * self.std + self.mean
        return value.astype(self.dtype)


def read_image(image_path, size=(128, 128), train=False):
    image = cv2.imread(image_path)
    if train:
        h, w, c = image.shape
        hc, wc = h // 2, w // 2
        M = cv2.getRotationMatrix2D((hc, wc), np.random.randint(360), np.random.uniform(1.3, 2.5))
        image = cv2.warpAffine(image, M, (h, w))
    image = cv2.resize(image, size)
    return image


def read_images(images, size=(128, 128)):
    images = [read_image(image, size) for image in images]
    return np.array(images)


class Datasets:
    def __init__(self, batch=32):
        cwd = path.split(__file__)[0]
        self.scaler = StandardScaler()
        self.horses = glob.glob('%s/trainA/*.jpg' % cwd)
        self.zebras = glob.glob('%s/trainB/*.jpg' % cwd)
        self.test_horses = glob.glob('%s/testA/*.jpg' % cwd)
        self.test_zebras = glob.glob('%s/testB/*.jpg' % cwd)
        self.batch = batch
        self.nums = max(len(self.horses), len(self.zebras))
        self.nums = math.ceil(self.nums / self.batch)
        self.index = 0
        self.epoch = 1

    def __iter__(self):
        print('开始第%d伦迭代' % self.epoch)
        return self

    def convert_data(self, image):
        image = self.scaler.inverse_transform(image)
        return image

    def __next__(self):
        if self.nums == self.index:
            self.epoch += 1
            self.index = 0
            print('开始第%d轮迭代' % self.epoch)
        self.index += 1
        self.horses = np.roll(self.horses, -self.batch, 0)
        self.zebras = np.roll(self.zebras, -self.batch, 0)
        horses = read_images(self.horses[:self.batch])
        horses = self.scaler.fit_transform(horses)
        zebras = read_images(self.zebras[:self.batch])
        zebras = self.scaler.fit_transform(zebras)
        return horses, zebras

    def sample_image(self, data_paths, batch):
        random.seed(None)
        horses = random.sample(data_paths, batch)
        data = []
        for horse in horses:
            horse = read_image(horse, train=False)
            data.append(horse)
        data = np.array(data)
        return data

    def load_test_dataset(self, batch=10):
        return self.sample_image(self.test_horses, batch), \
               self.sample_image(self.test_zebras, batch)


if __name__ == '__main__':
    datasets = Datasets(32)
    horse, zebra = datasets.load_test_dataset(1)
    print(horse.shape)
    for horse, zebra in datasets:
        image = np.concatenate((horse, zebra), axis=2)
        cv2.imshow('horse@zebra', image[0])
        cv2.waitKey(10)
