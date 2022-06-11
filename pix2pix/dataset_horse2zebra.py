import cv2
import os.path as path
import glob
import numpy as np
import random
from skimage.transform import resize, rotate

data_dir = '../horse2zebra'
data_save_dir = './horse2zebra'
label_trainA = 'trainA'
label_trainB = 'trainB'
label_testA = 'testA'
label_testB = 'testB'


def resize_train_data(data, size):
    if np.random.uniform() > .5:
        return np.array([cv2.resize(img, (size, size)) for img in list(data)])
    ix = np.random.randint(size)
    return data[:, ix:ix + size, ix:ix + size]


def resize_data(data, size):
    images = []
    for img in list(data):
        h, w, _ = img.shape
        if np.random.uniform() > .6:
            angle = np.random.uniform(360)
            rm = cv2.getRotationMatrix2D((h // 2, w // 2), angle, np.random.uniform(.75, 1.25))
            img = cv2.warpAffine(img, rm, (w, h))
        if np.random.uniform() > .5 or h < size or w < size:
            img = cv2.resize(img, (size, size))
        else:
            h, w, _ = img.shape
            hh, hw = (h - size) // 2, (w - size) // 2
            img = img[hh:hh + size, hw:hw + size]
        images.append(img)
    return np.array(images)


def transform_image(img, size):
    img = rotate(img, np.random.randint(360))
    scale = np.random.uniform(.75, 1.25)
    h, w, _ = img.shape
    h, w = int(scale * h), int(scale * w)
    img = resize(img, (h, w))
    hh = h // 2 - size // 2
    return img[hh:hh + size, hh:hh + size]


def image2numpy(label):
    data_save_path = f'{data_save_dir}/{label}.npy'
    if path.exists(data_save_path):
        return np.load(data_save_path, allow_pickle=True)
    data_paths = glob.glob(f'{data_dir}/{label}/*.jpg')
    data = [cv2.imread(data_path) for data_path in data_paths]
    np.save(data_save_path, data)
    return np.array(data)


class Dataset:
    def __init__(self, batch_size=1, resize=True):
        self.batch_size = batch_size
        self.trainA = image2numpy(label_trainA)
        self.trainB = image2numpy(label_trainB)
        self.testA = None
        self.testB = None
        self.scale_size = [i * 64 for i in range(1, 5)] if resize else None
        self.min_test_length = None

    def __iter__(self):
        np.random.seed(None)
        np.random.shuffle(self.trainA)
        np.random.shuffle(self.trainB)
        return self

    def roll_data(self):
        self.trainA, self.trainB = (np.roll(data, self.batch_size, 0) for data in (self.trainA, self.trainB))

    def __next__(self):
        self.roll_data()
        if self.scale_size is None:
            return self.trainA[:self.batch_size], self.trainB[:self.batch_size]
        scale_size = random.sample(self.scale_size, 1)
        scale_size = (scale_size[0], scale_size[0])
        trainA, trainB = [], []
        for i in range(self.batch_size):
            trainA.append(cv2.resize(self.trainA[i], scale_size, interpolation=cv2.INTER_AREA))
            trainB.append(cv2.resize(self.trainB[i], scale_size, interpolation=cv2.INTER_AREA))
        return np.array(trainA), np.array(trainB)

    def load_test_data(self, batch_size=1):
        self.testA = self.testA if self.testA is not None else image2numpy(label_testA)
        self.testB = self.testB if self.testB is not None else image2numpy(label_testB)
        self.min_test_length = self.min_test_length or min(len(self.testA), len(self.testB))
        idx = np.random.choice(self.min_test_length, batch_size, False)
        return self.testA[idx], self.testB[idx]


if __name__ == '__main__':
    dataset = Dataset(resize=False)
    testA, testB = dataset.load_test_data(1)
    testA, testB = resize_train_data(testA, 128), resize_train_data(testB, 128)
    cv2.imshow('A&B', np.hstack([testA[0], testB[0]]))
    cv2.waitKey()
    for trainA, trainB in dataset:
        trainA, trainB = resize_train_data(trainA, 128), resize_train_data(trainB, 128)
        cv2.imshow('A&B', np.hstack([trainA[0], trainB[0]]))
        cv2.waitKey(10)
