import tensorflow.keras as kr
from pix2pix_train import CycleGAN, load_data
from pix_models import RestNetPLayer, RestUNetPLayer

horse, zebra = [load_data(f'../horse2zebra', label, (64, 64)) for label in ('trainA', 'trainB')]
CycleGAN('horse2zebra13', horse, zebra,
         RestUNetPLayer((18, 3), 128, 3, last_layer=kr.layers.Conv2DTranspose(3, 1)),
         RestNetPLayer((9, 3), 128, 3, last_layer=kr.layers.Conv2D(1, 1, activation='sigmoid')))()
