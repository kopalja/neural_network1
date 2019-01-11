import numpy as np
import theano
import theano.tensor as T
from theano.tensor import shared_randomstreams
from Layer import Layer



class FullyConectedLayer(Layer):
    def __init__(self, in_size, out_size, activation_fn, learned_parameters = None): 
        Layer.__init__(self, [in_size, out_size, activation_fn])
        self.__in_size = in_size
        self.__activation_fn = activation_fn
        if learned_parameters == None:
            deviation = 1.0 / np.sqrt(in_size)
            if (activation_fn == T.nnet.softmax):
                deviation = 0
            self.__weights = theano.shared(value = np.random.normal(0.0, deviation, (in_size, out_size)).astype('float32'), borrow = True)
            self.__biases = theano.shared(value = np.random.normal(0.0, deviation, out_size).astype('float32'), borrow = True)
        else:
            self.__weights = theano.shared(value = np.asarray(learned_parameters[0], 'float32'), borrow = True)
            self.__biases = theano.shared(value = np.asarray(learned_parameters[1], 'float32'), borrow = True)
        self.params = [self.__weights, self.__biases]


    def __dropout(self, layer):
        srng = shared_randomstreams.RandomStreams(np.random.RandomState(0).randint(999999))
        np.random.random
        mask = srng.binomial(n = 1, p = 1.0 - Layer.dropout, size = layer.shape)
        return layer * T.cast(mask, theano.config.floatX)
    


    #__public__:

    def feed_forward(self, inpt):
        inpt = inpt.reshape((Layer.minibatch_size, self.__in_size))
        return self.__activation_fn((1.0 - Layer.dropout) * T.dot(inpt, self.__weights) + self.__biases)
    
    def feed_forward_dropout(self, inpt):
        inpt = inpt.reshape((Layer.minibatch_size, self.__in_size))
        inpt = self.__dropout(inpt)
        return self.__activation_fn(T.dot(inpt, self.__weights) + self.__biases)


