from pix2pix_train4 import CycleGAN
from pix_models import RestNet131, RestUNet131
from dataset_pix2pix import Dataset

CycleGAN('face2face9', Dataset('../horse2zebra'), RestUNet131(5, 128), RestNet131(5, 128))()
