import cv2
import numpy as np
import random


def read_images(video):
    vc = cv2.VideoCapture(video)
    images = []
    while True:
        ret, image = vc.read()
        if not ret:
            break
        image = cv2.resize(image, (128, 128))
        images.append(image / 255.)
    vc.release()
    return images


def convert(img):
    img *= 255.
    return img.astype(np.uint8)


class Datasets:
    def __init__(self, batch=1):
        self.horses = read_images('horse.mp4')
        self.zebras = read_images('zebra_Trim.mp4')
        self.batch = batch

    def __iter__(self):
        return self

    def __next__(self):
        horses = random.sample(self.horses, self.batch)
        zebras = random.sample(self.zebras, self.batch)
        return np.array(horses), np.array(zebras)


if __name__ == '__main__':
    datasets = Datasets()
    for horse, zebra in datasets:
        image = np.concatenate((horse, zebra), axis=2)
        cv2.imshow('horse@zebra', image[0])
        cv2.waitKey(10)
