import cv2
import os.path as path
import glob
import numpy as np
import random
from skimage.transform import resize

data_dir = '../horse2zebra/faces'
data_save_dir = './face2face'
label_trainA = 'trainB'
label_trainB = 'glz'


def image2numpy(label):
    data_save_path = f'{data_save_dir}/{label}.npy'
    if path.exists(data_save_path):
        return np.load(data_save_path, allow_pickle=True)
    data_paths = glob.glob(f'{data_dir}/{label}/*.jpg')
    data = [cv2.imread(data_path) for data_path in data_paths]
    np.save(data_save_path, data)
    return np.array(data)


class Dataset:
    def __init__(self, batch_size=1):
        self.batch_size = batch_size
        self.trainA = image2numpy(label_trainA)
        self.trainB = image2numpy(label_trainB)
        self.scale_size = [i * 8 for i in range(6, 32)]
        self.min_test_length = None

    def __iter__(self):
        np.random.seed(None)
        np.random.shuffle(self.trainA)
        np.random.shuffle(self.trainB)
        return self

    def roll_data(self):
        self.trainA, self.trainB = (np.roll(data, -self.batch_size, 0) for data in (self.trainA, self.trainB))

    def __next__(self):
        scale_size = random.sample(self.scale_size, 1)
        scale_size = (scale_size[0], scale_size[0])
        trainA, trainB = [], []
        for i in range(self.batch_size):
            trainA.append(resize(self.trainA[i], scale_size))
            trainB.append(resize(self.trainB[i], scale_size))
        self.roll_data()
        return np.array(trainA), np.array(trainB)

    def load_test_data(self, batch_size=1):
        self.min_test_length = self.min_test_length or min(len(self.trainA), len(self.trainB))
        idx = np.random.choice(self.min_test_length, batch_size, False)
        return self.trainA[idx], self.trainB[idx]


if __name__ == '__main__':
    dataset = Dataset()
    testA, testB = dataset.load_test_data(1)
    cv2.imshow('A&B', np.hstack([testA[0], testB[0]]))
    cv2.waitKey()
    for trainA, trainB in dataset:
        cv2.imshow('A&B', np.hstack([trainA[0], trainB[0]]))
        cv2.waitKey(10)
