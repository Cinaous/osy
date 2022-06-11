from pix2pix.pix_scaler import FsScaler as Scaler
from pix2pix.pix_models import RestNetModel, RestUNetModel, CycleNet, DiscriminatorNet
import tensorflow.keras as kr
import numpy as np
import cv2
import os.path as path


def sampling_img(img, size, n=2):
    h, w, _ = img.shape
    new_h = np.random.randint(h // n, h + 1)
    new_w = np.random.randint(w // n, w + 1)
    ix_h = np.random.randint(h - new_h + 1)
    ix_w = np.random.randint(w - new_w + 1)
    img = img[ix_h:ix_h + new_h, ix_w:ix_w + new_w]
    return cv2.resize(img, size)


if __name__ == '__main__':
    imgA = cv2.imread('../pix2pix/train_data/trainA/1.jpg')
    imgB = cv2.imread('../pix2pix/train_data/trainB/4.jpg')
    size = (64, 112)
    imgA, imgB = cv2.resize(imgA, size), cv2.resize(imgB, size)

    A2B = RestUNetModel((9, 3), 48, 3, last_layer=kr.layers.Conv2D(3, 1))
    B2A = RestUNetModel((9, 3), 48, 3, last_layer=kr.layers.Conv2D(3, 1))
    DA = RestNetModel((6, 3), 48, 3, last_layer=kr.layers.Conv2D(1, 1))
    DB = RestNetModel((6, 3), 48, 3, last_layer=kr.layers.Conv2D(1, 1))
    CNet = CycleNet(A2B, B2A, DA, DB, lambdas=[10, 10, 1, 1])
    DNet = DiscriminatorNet(A2B, B2A, DA, DB)
    lr = 2e-4
    label = 'cyq'
    mds = [CNet, DNet]
    md_save_path = [f'mds/{label}-{i + 1}' for i in range(2)]
    # for md, mp in zip(mds, md_save_path):
    #     if path.exists(mp + '.index'):
    #         md.load_weights(mp)
    scale = Scaler()
    epoch = 0
    train_shape = DA.compute_output_shape([1, *imgA.shape])
    real_label = np.ones(train_shape)
    fake_label = np.zeros_like(real_label)
    while True:
        trainA = sampling_img(imgA, size)
        trainB = sampling_img(imgB, size)
        trainA, trainB = scale.fit_transform([trainA]), scale.fit_transform([trainB])
        DNet.fit([trainA, trainB], [real_label, real_label, fake_label, fake_label], workers=12,
                 use_multiprocessing=True)
        CNet.fit([trainA, trainB], [trainA, trainB, real_label, real_label], workers=12,
                 use_multiprocessing=True)
        print(f'current epoch is {epoch}:')
        if epoch % 25 == 0:
            [md.save_weights(mp) for md, mp in zip(mds, md_save_path)]
            testA = scale.fit_transform([imgA])
            ta2b = A2B(testA)
            img1 = scale.inverse_transform(ta2b)[0]
            testB = scale.fit_transform([imgB])
            tb2a = B2A(testB)
            img2 = scale.inverse_transform(tb2a)[0]
            img = np.hstack((img1, imgA, img2, imgB))
            cv2.imwrite(f'images/{label}/train-{epoch}.jpg', img)
        epoch += 1
