import numpy as np
import theano
from theano.tensor.signal import pool
from Layer import Layer


class PoolLayer(Layer):
    def __init__(self, size):
        Layer.__init__(self, [size])
        self.size = (size, size)
        self.w = theano.shared(np.asarray(np.random.normal(0.0, 0, 0), theano.config.floatX), borrow=True)
        self.b = theano.shared(np.asarray(np.random.normal(0.0, 0, 0), theano.config.floatX), borrow=True)
        self.params = []

    def feed_forward(self, inpt):
        return pool.pool_2d(input = inpt, ws = self.size, ignore_border = True)

    def feed_forward_dropout(self, inpt):
        return self.feed_forward(inpt)
