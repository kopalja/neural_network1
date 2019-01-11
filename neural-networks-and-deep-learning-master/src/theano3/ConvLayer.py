import numpy as np
import theano
import theano.tensor as T
from Layer import Layer




class ConvLayer(Layer):
    def __init__(self, input_shape, output_images, kernel_size, activation_fn, learned_parameters = None): 
        Layer.__init__(self, [input_shape, output_images, kernel_size, activation_fn])
        self.__input_shape = input_shape
        self.__filter_shape = (output_images, input_shape[0], kernel_size, kernel_size)
        self.__activation_fn = activation_fn
        if learned_parameters == None:
            n_out = np.prod(self.__filter_shape) / (output_images * 4)
            self.__weights = theano.shared(value = np.random.normal(0.0, 1.0 / np.sqrt(n_out), self.__filter_shape).astype('float32'), borrow = True)
            self.__biases = theano.shared(value = np.random.normal(0.0, 1.0, output_images).astype('float32'), borrow = True)
        else:
            self.__weights = theano.shared(value = np.asarray(learned_parameters[0], 'float32'), borrow = True)
            self.__biases = theano.shared(value = np.asarray(learned_parameters[1], 'float32'), borrow = True)          
        self.params = [self.__weights, self.__biases]  


    #__public__: 
    
    def feed_forward(self, inpt):
        inpt_shape = (Layer.minibatch_size, self.__input_shape[0], self.__input_shape[1], self.__input_shape[2])
        inpt = inpt.reshape(inpt_shape)
        conv_output = theano.tensor.nnet.conv2d(input = inpt, filters = self.__weights, filter_shape = self.__filter_shape, input_shape = inpt_shape)
        return self.__activation_fn(conv_output + self.__biases.dimshuffle('x', 0, 'x', 'x'))
    
    def feed_forward_dropout(self, inpt):
        """ No dropout in convolution """
        return self.feed_forward(inpt)

    







    # def __batch_conv_normalize(self, conv_output):
    #     mean = T.mean(conv_output, axis = [0, 2, 3])
    #     std = T.std(conv_output, axis = [0, 2, 3])
    #     norm = (conv_output - mean.dimshuffle('x', 0, 'x', 'x')) / std.dimshuffle('x', 0, 'x', 'x')
    #     return self.y.dimshuffle('x', 0, 'x', 'x') * norm   