from pix2pix_train import CycleGAN, load_data
from pix_models import RestNet131, RestUNet131

horse, zebra = [load_data(f'../horse2zebra/faces', label, (64, 64)) for label in ('trainA', 'trainB')]
CycleGAN('face2face7', horse, zebra, RestUNet131(6, 96), RestNet131(6, 96))()
