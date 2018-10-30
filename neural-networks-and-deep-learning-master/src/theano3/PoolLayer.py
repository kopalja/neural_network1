import numpy as np
import theano

from theano.tensor.nnet import conv2d
from theano.tensor.signal import pool

from theano.tensor.nnet import conv



class PoolLayer:
    def __init__(self, size : tuple):
        self.size = size
        self.w = []
        self.b = []

    def feed_forward(self, inpt):
        return pool.pool_2d(input = inpt, ws = self.size, ignore_border = True)