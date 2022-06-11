import abc
import tensorflow.keras as kr
from pix2pix.pix_models import RestNet131
import tensorflow as tf

Net = RestNet131(3, 96)


class GenModel(kr.Model, abc.ABC):
    def __init__(self, A2B, B2A):
        super(GenModel, self).__init__()
        self.A2B = A2B
        self.B2A = B2A

    def call(self, inputs, training=None, mask=None):
        a, b = inputs
        a2b, b2a = self.A2B(a), self.B2A(b)
        return a2b, b2a, self.B2A(a2b), self.A2B(b2a), self.B2A(a), self.A2B(b)