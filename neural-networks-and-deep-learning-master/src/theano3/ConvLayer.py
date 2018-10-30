import numpy as np
import theano

from theano.tensor.nnet import conv2d
from theano.tensor.signal import pool

from theano.tensor.nnet import conv

import GlobalProperties


class ConvLayer:
    def __init__(self, input_shape, output_images, kernel_size, activation_fn):
        self.input_shape = (GlobalProperties.mini_batch_size, input_shape[0], input_shape[1], input_shape[2])
        self.filter_shape = (output_images, self.input_shape[1], kernel_size, kernel_size)
        self.activation_fn = activation_fn
        n_out = np.prod(self.filter_shape) / (self.filter_shape[1] * 4)
        self.w = theano.shared(np.asarray(np.random.normal(0.0, np.sqrt(1.0 / n_out), self.filter_shape), theano.config.floatX), borrow=True)
        self.b = theano.shared(np.asarray(np.random.normal(0.0, 1.0, output_images), theano.config.floatX), borrow=True)
        
    def feed_forward(self, inpt):
        inpt = inpt.reshape(self.input_shape)
        conv_output = conv2d(input = inpt, filters = self.w, filter_shape = self.filter_shape, input_shape = self.input_shape)
        return self.activation_fn(conv_output + self.b.dimshuffle('x', 0, 'x', 'x'))