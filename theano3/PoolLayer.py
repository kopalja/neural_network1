import numpy as np
import theano
from theano.tensor.signal import pool
from Layer import Layer


class PoolLayer(Layer):
    def __init__(self, shape, mode = 'max'): 
        Layer.__init__(self, [shape, mode])
        self.__shape = shape
        self.params = []

    def feed_forward(self, inpt):
        return pool.pool_2d(input = inpt, ws = self.__shape, ignore_border = True, mode='average_inc_pad')

    def feed_forward_dropout(self, inpt):
        return self.feed_forward(inpt)


