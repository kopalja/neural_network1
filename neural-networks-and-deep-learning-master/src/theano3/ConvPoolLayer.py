import numpy as np
import theano

from theano.tensor.nnet import conv2d
from theano.tensor.signal import pool

class ConvPoolLayer:
    def __init__(self, input_shape, output_images, kernel_size, pool_shape, activation_fn):
        self.input_shape = input_shape
        self.filter_shape = (output_images, input_shape[1], kernel_size, kernel_size)
        self.pool_shape = pool_shape
        self.activation_fn = activation_fn
        n_out = np.prod(self.filter_shape) / (np.prod(pool_shape) * input_shape[1])
        self.w = theano.shared(np.asarray(np.random.normal(0.0, np.sqrt(1 / n_out), self.filter_shape), theano.config.floatX))
        self.b = theano.shared(np.asarray(np.random.normal(0.0, 1.0, output_images), theano.config.floatX))
    
    def feed_forward(self, inpt):
        inpt = inpt.reshape(self.input_shape)
        conv_output = conv2d(input = inpt, filters = self.w, filter_shape = self.filter_shape, input_shape = self.input_shape)
        pool_output = pool.pool_2d(input = conv_output, ws = self.pool_shape, ignore_border = True)
        return self.activation_fn(pool_output + self.b.dimshuffle('x', 0, 'x', 'x'))



class ConvLayer:
    def __init__(self, input_shape, output_images, kernel_size, activation_fn):
        self.input_shape = input_shape
        self.filter_shape = (output_images, input_shape[1], kernel_size, kernel_size)
        self.activation_fn = activation_fn
        n_out = np.prod(self.filter_shape) / input_shape[1]
        self.w = theano.shared(np.asarray(np.random.normal(0.0, np.sqrt(1 / n_out),self.filter_shape), theano.config.floatX))
        self.b = theano.shared(np.asarray(np.random.normal(0.0, 1.0, output_images), theano.config.floatX))
    
    def feed_forward(self, inpt):
        inpt = inpt.reshape(self.input_shape)
        conv_output = conv2d(input = inpt, filters = self.w, filter_shape = self.filter_shape, input_shape = self.input_shape)
        return self.activation_fn(conv_output + self.b.dimshuffle('x', 0, 'x', 'x'))