import glob
import cv2
import numpy as np
import os.path as path
import tensorflow as tf


class Datasets:
    def __init__(self):
        self.trainA_dir = 'trainA/*.jpg'
        self.trainA_save_path = 'horse_train.npy'
        self.trainB_dir = 'trainB/*.jpg'
        self.trainB_save_path = 'zebra_train.npy'
        self.testA_dir = 'testA/*.jpg'
        self.testA_save_path = 'horse_test.npy'
        self.testB_dir = 'testB/*.jpg'
        self.testB_save_path = 'zebra_test.npy'

    def load_train_dataset(self):
        horses = self.load_dataset(self.trainA_dir, self.trainA_save_path)
        zebras = self.load_dataset(self.trainB_dir, self.trainB_save_path)
        return horses, zebras

    def load_test_dataset(self):
        horses = self.load_dataset(self.testA_dir, self.testA_save_path)
        zebras = self.load_dataset(self.testB_dir, self.testB_save_path)
        return horses, zebras

    def train_db(self, batch_size=32):
        return tf.data.Dataset.from_tensor_slices(self.zip(*self.load_train_dataset())).batch(batch_size)

    def test_db(self, batch_size=32):
        return tf.data.Dataset.from_tensor_slices(self.zip(*self.load_test_dataset())).batch(batch_size)

    @staticmethod
    def zip(x, y):
        lx, ly = len(x), len(y)
        if lx < ly:
            x = Datasets.fill(x, ly)
        elif lx > ly:
            y = Datasets.fill(y, lx)
        return x, y

    @staticmethod
    def fill(a, mx):
        la = len(a)
        # for ix in range(la, mx):
        #     a = np.append(a, [a[(mx - ix) % la]], axis=0)
        a = np.append(a, a[:mx - la], axis=0)
        return a

    @staticmethod
    def load_dataset(data_dir, data_save_path):
        if path.exists(data_save_path):
            return np.load(data_save_path)
        imgs = [cv2.imread(path) for path in glob.glob(data_dir)]
        imgs = np.array(imgs) / 255. - .5
        np.save(data_save_path, imgs)
        return np.array(imgs)

    @staticmethod
    def convert_data(data):
        data: np.ndarray = (data + .5) * 255.
        return data.astype(np.uint8)


if __name__ == '__main__':
    dataset = Datasets()
