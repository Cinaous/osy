import random
import cv2
import numpy as np
import cv2 as cv
import glob


class Dataset:
    def __init__(self, test_num=30, batch_size=1):
        super(Dataset, self).__init__()
        image_dir = 'D:/workspace/oyhcrawler/classic'
        self.beauties = glob.glob(f'{image_dir}/1/*.jpg')
        self.normals = glob.glob(f'{image_dir}/2/*.jpg')
        self.beauties = self.read_image(self.beauties)
        self.normals = self.read_image(self.normals)
        np.random.seed(116)
        np.random.shuffle(self.beauties)
        np.random.shuffle(self.normals)
        self.beauties_train = self.beauties[:-test_num]
        self.beauties_test = self.beauties[-test_num:]
        self.normals_train = self.normals[:-test_num]
        self.normals_test = self.normals[-test_num:]
        self.batch_size = batch_size
        self.test_num = test_num

    def read_image(self, data_paths):
        imgs = []
        for image_path in data_paths:
            img = cv.imread(image_path)
            img = cv2.resize(img, (128, 128)) / 255.
            imgs.append(img)
        return imgs

    def load_test(self, batch_size=None):
        batch_size = batch_size or self.test_num
        beauties = random.sample(self.beauties_test, batch_size)
        normals = random.sample(self.normals_test, batch_size)
        return beauties, np.zeros(batch_size), normals, np.ones(batch_size)

    def __iter__(self):
        np.random.shuffle(self.beauties_train)
        np.random.shuffle(self.normals_train)
        return self

    def __next__(self):
        beauties = random.sample(self.beauties_train, self.batch_size)
        normals = random.sample(self.normals_train, self.batch_size)
        return np.array(beauties), np.zeros(len(beauties)), np.array(normals), np.ones(len(normals))


if __name__ == '__main__':
    dataset = Dataset(batch_size=1)
    for beauty, label_0, normal, label_1 in dataset:
        img = np.hstack([beauty[0], normal[0]])
        cv.imshow('beauties', img)
        cv.waitKey(10)
