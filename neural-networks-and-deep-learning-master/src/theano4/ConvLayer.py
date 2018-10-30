import theano
import numpy as np
import theano.tensor as T


from theano.tensor.nnet import conv2d
from theano.tensor.signal import pool

class ConvPoolLayer:
    def __init__(self, filter_shape, image_shape, poolsize, activation_fn):
        self.filter_shape = filter_shape
        self.image_shape = image_shape
        self.poolsize = poolsize
        self.activation_fn=activation_fn
        # initialize weights and biases
        n_out = (filter_shape[0]*np.prod(filter_shape[2:])/np.prod(poolsize))
        self.w = theano.shared(np.asarray(np.random.normal(loc=0, scale=np.sqrt(1.0/n_out), size=filter_shape), dtype=theano.config.floatX), borrow=True)
        self.b = theano.shared(np.asarray(np.random.normal(loc=0, scale=1.0, size=(filter_shape[0],)), dtype=theano.config.floatX), borrow=True)
        self.params = [self.w, self.b]

    def set_inpt(self, inpt):
        self.inpt = inpt.reshape(self.image_shape)
        conv_out = conv2d(input=self.inpt, filters=self.w, filter_shape=self.filter_shape, input_shape=self.image_shape)
        pooled_out = pool.pool_2d(input=conv_out, ds=self.poolsize, ignore_border=True)
        self.output = self.activation_fn(pooled_out + self.b.dimshuffle('x', 0, 'x', 'x'))


