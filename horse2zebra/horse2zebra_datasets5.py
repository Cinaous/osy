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
        image = cv2.resize(image, (256, 256)) / 255.
        images.append(image)
    vc.release()
    return images


class Datasets:
    def __init__(self, batch=1):
        self.horses = read_images('horse.mp4')
        self.zebras = read_images('zebra_Trim.mp4')
        self.batch = batch

    def __iter__(self):
        return self

    @staticmethod
    def convert_data(image):
        image: np.ndarray = image * 255.
        return image.astype('uint8')

    def __next__(self):
        return np.array(random.sample(self.horses, self.batch)), np.array(random.sample(self.zebras, self.batch))

    def load_test_dataset(self, batch=10):
        return np.array(random.sample(self.horses, batch)), np.array(random.sample(self.zebras, batch))


if __name__ == '__main__':
    datasets = Datasets()
    horse, zebra = datasets.load_test_dataset()
    print(horse.shape)
    for horse, zebra in datasets:
        image = np.concatenate((horse, zebra), axis=2)
        cv2.imshow('horse@zebra', image[0])
        cv2.waitKey(10)
