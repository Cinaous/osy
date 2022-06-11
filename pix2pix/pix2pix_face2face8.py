from pix2pix_train3 import CycleGAN, load_data
from pix_models import RestNet131, RestUNet131
from dataset2_pix2pix import Dataset

CycleGAN('face2face8', Dataset('../horse2zebra'), RestUNet131(5, 96), RestNet131(5, 96))()
