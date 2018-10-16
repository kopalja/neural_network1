import numpy as np
import theano
import theano.tensor as T


class FullyConectedLayer(object):
    def __init__(self, input_shape, out_size, activation_fn):
        self.input_shape = input_shape
        self.activation_fn = activation_fn
        self.w = theano.shared(np.asarray(np.random.normal(0.0, np.sqrt(1.0 / out_size), size = (input_shape[1], out_size)), theano.config.floatX))
        self.b = theano.shared(np.asarray(np.random.normal(0.0, 1.0, out_size), theano.config.floatX))

    def feed_forward(self, inpt):
        inpt = inpt.reshape(self.input_shape)
        return self.activation_fn(T.dot(inpt, self.w) + self.b)


    