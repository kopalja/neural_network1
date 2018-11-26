import numpy as np
import theano
import theano.tensor as T
from Layer import Layer


class ConvLayer(Layer):
    def __init__(self, input_shape, output_images, kernel_size, activation_fn, learned_parameters = None):
        Layer.__init__(self, [input_shape, output_images, kernel_size, activation_fn])
        self.input_shape = input_shape
        self.filter_shape = (output_images, input_shape[0], kernel_size, kernel_size)
        self.activation_fn = activation_fn
        if learned_parameters == None:
            n_out = np.prod(self.filter_shape) / (output_images * 4)
            self._w = theano.shared(np.asarray(np.random.normal(0.0, 1.0 / np.sqrt(n_out), self.filter_shape), theano.config.floatX), borrow=True)
            self._b = theano.shared(np.asarray(np.random.normal(0.0, 1.0, output_images), theano.config.floatX), borrow=True)
        else:
            self._w = theano.shared(np.asarray(learned_parameters[0], theano.config.floatX), borrow = True)
            self._b = theano.shared(np.asarray(learned_parameters[1], theano.config.floatX), borrow = True)          
        self.params = [self._w, self._b]  


    #__public__: 
    
    def feed_forward(self, inpt):
        inpt_shape = ((Layer.minibatch_size, self.input_shape[0], self.input_shape[1], self.input_shape[2]))
        inpt = inpt.reshape(inpt_shape)
        conv_output = theano.tensor.nnet.conv2d(input = inpt, filters = self._w, filter_shape = self.filter_shape, input_shape = inpt_shape)
        return self.activation_fn(conv_output + self._b.dimshuffle('x', 0, 'x', 'x'))
    
    # no dropout in convolution
    def feed_forward_dropout(self, inpt):
        return self.feed_forward(inpt)








    # def __batch_conv_normalize(self, conv_output):
    #     mean = T.mean(conv_output, axis = [0, 2, 3])
    #     std = T.std(conv_output, axis = [0, 2, 3])
    #     norm = (conv_output - mean.dimshuffle('x', 0, 'x', 'x')) / std.dimshuffle('x', 0, 'x', 'x')
    #     return self.y.dimshuffle('x', 0, 'x', 'x') * norm   