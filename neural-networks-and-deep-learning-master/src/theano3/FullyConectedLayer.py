import numpy as np
import theano
import theano.tensor as T

import GlobalProperties

class FullyConectedLayer(object):
    def __init__(self, in_size, out_size, activation_fn):
        self.input_shape = (GlobalProperties.mini_batch_size, in_size)
        self.activation_fn = activation_fn
        deviation = np.sqrt(1.0 / out_size)
        if (activation_fn == T.nnet.softmax):
            deviation = 0
        self.w = theano.shared(np.asarray(np.random.normal(0.0, deviation, (self.input_shape[1], out_size)), theano.config.floatX), borrow=True)
        self.b = theano.shared(np.asarray(np.random.normal(0.0, deviation, out_size), theano.config.floatX), borrow=True)

    def feed_forward(self, inpt):
        inpt = inpt.reshape(self.input_shape)
        return self.activation_fn(T.dot(inpt, self.w) + self.b)

    