import cv2
import numpy as np
import random
import glob
from horse2zebra.horse2zebra_datasets12 import StandardScaler

scaler = StandardScaler()


class Datasets:
    def __init__(self, batch=32):
        self.horses = glob.glob('trainA/*.jpg')
        self.zebras = glob.glob('trainB/*.jpg')
        self.test_horses = glob.glob('testA/*.jpg')
        self.test_zebras = glob.glob('testB/*.jpg')
        self.batch = batch

    def __iter__(self):
        return self

    def __next__(self):
        return self.sample_image(self.horses, self.batch, True), self.sample_image(self.zebras, self.batch, True)

    @staticmethod
    def sample_image(data_paths, batch, train=False):
        random.seed(None)
        horses = random.sample(data_paths, batch)
        data = []
        for horse in horses:
            horse = cv2.imread(horse)
            horse = cv2.resize(horse, (128, 128))
            if train:
                m = cv2.getRotationMatrix2D((64, 64),
                                            np.random.randint(360),
                                            np.random.uniform(.3, 3))
                horse = cv2.warpAffine(horse, m, (128, 128))
                horse = scaler.fit_transform(horse)
            data.append(horse)
        return np.array(data)

    def load_test_dataset(self, batch=10):
        return self.sample_image(self.test_horses, batch), self.sample_image(self.test_zebras, batch)


if __name__ == '__main__':
    datasets = Datasets()
    horse, zebra = datasets.load_test_dataset()
    print(horse.shape)
    for horse, zebra in datasets:
        image = np.concatenate((horse, zebra), axis=2)
        cv2.imshow('horse@zebra', image[0])
        cv2.waitKey(10)
