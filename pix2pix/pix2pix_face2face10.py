from pix2pix_train4 import CycleGAN
from pix_models import RestNetModel, RestUNetModel
from dataset_pix2pix import Dataset
import tensorflow.keras as kr

CycleGAN('face2face10', Dataset('../horse2zebra'),
         RestUNetModel((18, 3), 128, 3, last_layer=kr.layers.Conv2DTranspose(3, 3, padding='same', activation='tanh')),
         RestNetModel((9, 3), 128, 3, last_layer=kr.layers.Conv2D(1, 3, padding='same', activation='sigmoid')))()
