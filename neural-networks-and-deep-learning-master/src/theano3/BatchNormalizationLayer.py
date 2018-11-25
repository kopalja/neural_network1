import numpy as np
import theano
import theano.tensor as T
from Layer import Layer

class BatchNormalizationLayer(Layer):
    def __init__(self, in_size):
        self.input_shape = (Layer.minibatch_size, in_size)
        self.w = theano.shared(np.asarray(np.random.normal(0.0, 1, (in_size)), theano.config.floatX), borrow=True)
        self.b = theano.shared(np.asarray(np.random.normal(0.0, 1, (in_size)), theano.config.floatX), borrow=True)
        self.params = [self.w, self.b]

    def feed_forward(self, inpt):
        inpt = inpt.reshape(self.input_shape)
        mean = T.mean(inpt, axis = 0)
        std = T.std(inpt, axis = 0)
        norm = (inpt - mean) / std
        return self.w * norm + self.b

    def feed_forward_dropout(self, inpt):
        return self.feed_forward(inpt)