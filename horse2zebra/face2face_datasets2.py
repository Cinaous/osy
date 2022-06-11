import glob
import cv2
import numpy as np
import random


def read_images(video):
    imgs_path = glob.glob(video)
    return imgs_path


def read_image(img_path):
    img = cv2.imread(img_path)
    return img / 255.


def convert(img: np.ndarray):
    img *= 255.
    return img.astype(np.uint8)


def rotation(img):
    h, w, _ = img.shape
    center = (h / 2, w / 2)
    m = cv2.getRotationMatrix2D(center, np.random.randint(360), np.random.uniform(.3, 3))
    return cv2.warpAffine(img, m, (h, w))


pools, pool_size = [], 100


class Datasets:
    def __init__(self, batch=1):
        horses = read_images('faces/trainB/*.jpg')
        self.horses = [read_image(horse) for horse in horses]
        zebras = read_images('faces/trainA/*.jpg')
        self.zebras = [read_image(zebra) for zebra in zebras]
        self.batch = batch or min(len(self.horses), len(self.zebras))

    def __iter__(self):
        return self

    def __next__(self):
        horses = random.sample(self.horses, self.batch)
        zebras = random.sample(self.zebras, self.batch)
        # horses = [rotation(horse) for horse in horses]
        # zebras = [rotation(zebra) for zebra in zebras]
        horses, zebras = np.array(horses), np.array(zebras)
        return horses, zebras

    def load_test_dataset(self, batch=10):
        horses = random.sample(self.horses, batch)
        zebras = random.sample(self.zebras, batch)
        return np.array(horses), np.array(zebras)


if __name__ == '__main__':
    datasets = Datasets()
    horse, zebra = datasets.load_test_dataset()
    print(horse.shape)
    for horse, zebra in datasets:
        image = np.concatenate((horse, zebra), axis=2)
        cv2.imshow('horse@zebra', image[0])
        cv2.waitKey(10)
