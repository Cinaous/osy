import cv2
import numpy as np
from face2face_datasets1 import Datasets
from models_model.model_models9 import UNet, Discriminator, GCModel

batch_size = 1
dataset = Datasets(batch=batch_size)
Z2H = UNet([64, 128, 256, 512, 512, 512, 512])
H2Z = UNet([64, 128, 256, 512, 512, 512, 512])
DH = Discriminator([64, 128, 256, 512])
DZ = Discriminator([64, 128, 256, 512])

GC = GCModel(H2Z, Z2H, DH, DZ)

GC_save_path = 'outputs/GC12.ckpt'
print('load GC weights...')
GC.load_weights(GC_save_path)

for test_horse, test_zebra in dataset:
    h2z = H2Z(test_horse)
    h2z2h = Z2H(h2z)
    img1 = np.concatenate((test_horse, h2z, h2z2h), axis=2)[0]
    z2h = Z2H(test_zebra)
    z2h2z = H2Z(z2h)
    img2 = np.concatenate((test_zebra, z2h, z2h2z), axis=2)[0]
    img3 = np.vstack((img1, img2))
    img3 = Datasets.convert_data(img3)
    cv2.imshow('trainA&trainB', img3)
    cv2.waitKey()
    cv2.destroyAllWindows()
