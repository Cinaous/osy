import cv2
import glob
import numpy as np
import random
import os.path as path
from pix_scaler import MsScaler


class Dataset:
    def __init__(self, data_dir, size=64, batch_size=1, scale=2):
        self.batch_size = batch_size
        self.output_shape = [batch_size, size, size, 3]
        self.scale = scale
        self.ms = MsScaler()
        self.data0_save_path = f'{data_dir}/data-{size}-0.npy'
        self.data1_save_path = f'{data_dir}/data-{size}-1.npy'
        self.size = size
        self.imsize = None
        if not self.read_numpy():
            self.trainA, self.trainB = [glob.glob(f'{data_dir}/{label}/*') for label in ('trainA', 'trainB')]
            self.read_image()
        self.ms.fit_transform(np.vstack([self.trainA, self.trainB]))
        self.trainA, self.trainB = list(self.ms.transform(self.trainA)), list(self.ms.transform(self.trainB))

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
        imgs = []
        h, w, _ = data[0].shape
        for img in data:
            if self.scale != 1:
                h_size = np.random.randint(h // self.scale, h + 1)
                w_size = np.random.randint(w // self.scale, w + 1)
                hx = np.random.randint(h - h_size + 1)
                wx = np.random.randint(w - w_size + 1)
                img = img[hx:hx + h_size, wx:wx + w_size]
            img = rotation_image(img, self.size)
            imgs.append(img)
        return np.array(imgs)

    def __next__(self):
        return self.sample(self.trainA), self.sample(self.trainB)


def rotation_image(img, size):
    h, w, _ = img.shape
    angle, sl = np.random.randint(360), max(h, w) / min(h, w)
    M = cv2.getRotationMatrix2D((w // 2, h // 2), angle,
                                sl * (abs(np.sin(np.pi * angle / 180)) + abs(np.cos(np.pi * angle / 180))))
    img = cv2.warpAffine(img, M, (w, h))
    return cv2.resize(img, (size, size))


if __name__ == '__main__':
    dataset = Dataset('../horse2zebra/faces', 128)
    testA, testB = dataset.load_test_data()
    scale = MrScaler()
    testA, testB = scale.fit_transform(testA / 255.), scale.fit_transform(testB / 255.)
    img1, img2 = testA[0], testB[0]
    img = np.concatenate([img1, img2], 1)
    cv2.imshow('img', img)
    cv2.waitKey()
    for trainA, trainB in dataset:
        trainA, trainB = scale.fit_transform(trainA / 255.), scale.fit_transform(trainB / 255.)
        print(np.max(trainA), np.min(trainA))
        img1, img2 = trainA[0], trainB[0]
        img = np.concatenate([img1, img2], 1)
        cv2.imshow('img', img)
        cv2.waitKey(10)
