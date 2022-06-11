import math
import cv2
import numpy as np
import random
import glob


def read_image(image_path, size=(128, 128)):
    image = cv2.imread(image_path)
    image = cv2.resize(image, size)
    image = image / 255.
    return image


def read_images(images, size=(128, 128)):
    images = [read_image(image, size) for image in images]
    return images


def convert(img):
    img = img * 255.
    return img.astype(np.uint8)


class Datasets:
    def __init__(self, batch=1):
        self.horses = glob.glob('trainA/*.jpg')
        self.zebras = glob.glob('trainB/*.jpg')
        self.horses = read_images(self.horses)
        self.zebras = read_images(self.zebras)
        self.batch = batch
        self.nums = max(len(self.horses), len(self.zebras))
        self.nums = math.ceil(self.nums / self.batch)
        self.index = 0
        self.epoch = 1

    def __iter__(self):
        print('开始第%d伦迭代' % self.epoch)
        return self

    def __next__(self):
        if self.nums == self.index:
            self.epoch += 1
            self.index = 0
            print('开始第%d轮迭代' % self.epoch)
        self.index += 1
        horses = np.array(random.sample(self.horses, self.batch))
        zebras = np.array(random.sample(self.zebras, self.batch))
        return horses, zebras

    def load_test_dataset(self, batch=10):
        test_horses = np.array(random.sample(self.horses, batch))
        test_zebras = np.array(random.sample(self.zebras, batch))
        return test_horses, test_zebras


if __name__ == '__main__':
    datasets = Datasets()
    horse, zebra = datasets.load_test_dataset()
    print(horse.shape)
    for horse, zebra in datasets:
        image = np.concatenate((horse, zebra), axis=2)
        image = convert(image)
        cv2.imshow('horse@zebra', image[0])
        cv2.waitKey(10)
