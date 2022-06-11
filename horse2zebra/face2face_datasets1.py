import glob
import cv2
import numpy as np
import random
from horse2zebra.horse2zebra_datasets12 import StandardScaler

scaler = StandardScaler()


def read_images(video):
    imgs_path = glob.glob(video)
    return imgs_path


def read_image(img_path, train=False):
    img = cv2.imread(img_path)
    if train:
        h, w, _ = img.shape
        center = (h / 2, w / 2)
        m = cv2.getRotationMatrix2D(center, np.random.randint(360), np.random.uniform(.3, 3))
        img = cv2.warpAffine(img, m, (h, w))
    return img


pools, pool_size = [], 100


def put_pools(obj):
    global pools
    if len(pools) == pool_size:
        pools = pools[1:]
    pools.append(obj)


def sample_pools():
    return random.sample(pools, 1)[0][0], random.sample(pools, 1)[0][1]


class Datasets:
    def __init__(self, batch=1):
        self.horses = read_images('faces/trainB/*.jpg')
        self.zebras = read_images('faces/trainA/*.jpg')
        self.batch = batch

    def __iter__(self):
        return self

    def __next__(self):
        horses, zebras = self.load_test_dataset(self.batch, True)
        return scaler.fit_transform(horses), scaler.fit_transform(zebras)

    def load_test_dataset(self, batch=10, train=False):
        random.seed(None)
        horses = random.sample(self.horses, batch)
        horses = [read_image(horse, train) for horse in horses]
        zebras = random.sample(self.zebras, batch)
        zebras = [read_image(zebra, train) for zebra in zebras]
        return np.array(horses), np.array(zebras)


if __name__ == '__main__':
    datasets = Datasets()
    horse, zebra = datasets.load_test_dataset()
    print(horse.shape)
    for horse, zebra in datasets:
        image = np.concatenate((horse, zebra), axis=2)
        cv2.imshow('horse@zebra', image[0])
        cv2.waitKey(10)
