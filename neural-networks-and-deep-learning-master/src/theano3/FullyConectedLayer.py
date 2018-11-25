import numpy as np
import theano
import theano.tensor as T
from theano.tensor import shared_randomstreams
from Layer import Layer



class FullyConectedLayer(Layer):
    def __init__(self, in_size, out_size, activation_fn):
        Layer.__init__(self, [in_size, out_size, activation_fn])
        
        self.input_params = (in_size, out_size, activation_fn)
        self.input_shape = in_size
        self.activation_fn = activation_fn
        deviation = 1.0 / np.sqrt(in_size)
        if (activation_fn == T.nnet.softmax):
            deviation = 0
        self.w = theano.shared(np.asarray(np.random.normal(0.0, deviation, (in_size, out_size)), theano.config.floatX), borrow=True)
        self.b = theano.shared(np.asarray(np.random.normal(0.0, deviation, out_size), theano.config.floatX), borrow=True)
        self.params = [self.w, self.b]


    def __dropout(self, layer):
        srng = shared_randomstreams.RandomStreams(np.random.RandomState(0).randint(999999))
        mask = srng.binomial(n = 1, p = 1.0 - Layer.dropout, size = layer.shape)
        return layer * T.cast(mask, theano.config.floatX)
    


    #__public__:

    def feed_forward(self, inpt):
        inpt = inpt.reshape((Layer.minibatch_size, self.input_shape))
        return self.activation_fn((1.0 - Layer.dropout) * T.dot(inpt, self.w) + self.b)
    
    def feed_forward_dropout(self, inpt):
        inpt = inpt.reshape((Layer.minibatch_size, self.input_shape))
        inpt = self.__dropout(inpt)
        return self.activation_fn(T.dot(inpt, self.w) + self.b)