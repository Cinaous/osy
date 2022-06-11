import cv2
import numpy as np
import random
import glob


class Datasets:
    def __init__(self, batch=32):
        self.horses = glob.glob('trainA/*.jpg')
        self.zebras = glob.glob('trainB/*.jpg')
        self.test_horses = glob.glob('testA/*.jpg')
        self.test_zebras = glob.glob('testB/*.jpg')
        self.batch = batch

    def __iter__(self):
        return self

    @staticmethod
    def convert_data(image):
        image: np.ndarray = (image + 1.) * 127.5
        return image.astype('uint8')

    def __next__(self):
        return self.sample_image(self.horses, self.batch), self.sample_image(self.zebras, self.batch)

    @staticmethod
    def sample_image(data_paths, batch, train=True):
        random.seed(None)
        horses = random.sample(data_paths, batch)
        data = []
        for horse in horses:
            horse = cv2.imread(horse)
            if train:
                h, w, _ = horse.shape
                center = (h / 2, w / 2)
                m = cv2.getRotationMatrix2D(center, np.random.randint(360), np.random.uniform(1.2, 3.5))
                horse = cv2.warpAffine(horse, m, (h, w))
            horse = horse / 127.5 - 1.
            data.append(horse)
        return np.array(data)

    def load_test_dataset(self, batch=10):
        return self.sample_image(self.test_horses, batch, False), self.sample_image(self.test_zebras, batch, False)


if __name__ == '__main__':
    datasets = Datasets()
    horse, zebra = datasets.load_test_dataset()
    print(horse.shape)
    for horse, zebra in datasets:
        image = np.concatenate((horse, zebra), axis=2)
        cv2.imshow('horse@zebra', image[0])
        cv2.waitKey(10)
