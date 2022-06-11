import cv2
import numpy as np


def load_data(show_image=False, image_size=(64, 36)):
    video_path = '../horse2zebra/glz.mp4'
    big_data, small_data = [], []
    capture = cv2.VideoCapture(video_path)
    ix = sx = 0
    while True:
        ret, image = capture.read()
        ix += 1
        print(f'current image is {ix} picture.', end='\r')
        if sx < 200:
            sx += 1
            continue
        if not ret:
            break
        if image_size is not None:
            image = cv2.resize(image, image_size, interpolation=cv2.INTER_AREA)
        if show_image:
            cv2.imshow('glz', image)
            cv2.waitKey(10)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        big_data.append(image)
        image = cv2.resize(image, (0, 0), fx=.5, fy=.5, interpolation=cv2.INTER_AREA)
        small_data.append(image)
    capture.release()
    if show_image:
        cv2.destroyAllWindows()
    return np.array(big_data), np.array(small_data)


class Dataset:
    def __init__(self, batch_size=1, show_image=False, image_size=(88, 50)):
        self.batch_size = batch_size
        self.big_data, self.small_data = load_data(show_image, image_size)
        self.data_length = len(self.big_data)

    def __iter__(self):
        np.random.seed(None)
        return self

    def __next__(self):
        idx = np.random.choice(self.data_length, self.batch_size, False)
        return self.small_data[idx], self.big_data[idx]
