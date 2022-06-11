import cv2
import glob
import numpy as np
import random
import os.path as path
from pix_scaler import FixedScaler as Scaler


class Dataset:
    def __init__(self, data_dir, size=64, batch_size=1):
        self.batch_size = batch_size
        self.output_shape = [batch_size, size, size, 3]
        self.data0_save_path = f'{data_dir}/data-{size}-0.npy'
        self.data1_save_path = f'{data_dir}/data-{size}-1.npy'
        self.size = size
        self.imsize = None
        if not self.read_numpy():
            self.trainA, self.trainB = [glob.glob(f'{data_dir}/{label}/*') for label in ('trainA', 'trainB')]
            self.read_image()

    def load_test_data(self, batch_size=1):
        return np.array(random.sample(self.trainA, batch_size)), np.array(random.sample(self.trainB, batch_size))

    def read_numpy(self):
        c = 0
        if path.exists(self.data0_save_path):
            c += 1
            self.trainA = list(np.load(self.data0_save_path, allow_pickle=True))
        if path.exists(self.data1_save_path):
            c += 1
            self.trainB = list(np.load(self.data1_save_path, allow_pickle=True))
        return c == 2

    def read_image(self):
        self.trainA = [self.img_numpy(img) for img in self.trainA]
        self.trainB = [self.img_numpy(img) for img in self.trainB]
        np.save(self.data0_save_path, self.trainA)
        np.save(self.data1_save_path, self.trainB)

    def img_numpy(self, img_path):
        img = cv2.imread(img_path)
        if self.imsize is not None:
            return cv2.resize(img, self.imsize)
        h, w, _ = img.shape
        ih, iw = h // self.size, w // self.size
        self.imsize = (iw * self.size, ih * self.size)
        return cv2.resize(img, self.imsize)

    def __iter__(self):
        return self

    def sample(self, data):
        data = random.sample(data, self.batch_size)
        data = [cv2.resize(img, (self.size, self.size)) for img in data]
        return np.array(data)

    def __next__(self):
        return self.sample(self.trainA), self.sample(self.trainB)


if __name__ == '__main__':
    dataset = Dataset('../horse2zebra')
    testA, testB = dataset.load_test_data()
    scale = Scaler()
    testA, testB = scale.fit_transform(testA), scale.fit_transform(testB)
    img1, img2 = testA[0], testB[0]
    img = np.concatenate([img1, img2], 1)
    cv2.imshow('img', img)
    cv2.waitKey()
    for trainA, trainB in dataset:
        trainA, trainB = scale.fit_transform(trainA), scale.fit_transform(trainB)
        print(np.max(trainA), np.min(trainA))
        img1, img2 = trainA[0], trainB[0]
        img = np.concatenate([img1, img2], 1)
        cv2.imshow('img', img)
        cv2.waitKey(10)
