import theano
import numpy as np
import theano.tensor as T

class FullyConnectedLayer(object):   
    def __init__(self, mini_batch_size, n_in, n_out, activation_fn, zerovs = False):
        self.mini_batch_size = mini_batch_size
        self.n_in = n_in
        self.n_out = n_out
        self.activation_fn = activation_fn
        # Initialize weights and biases

        self.w = theano.shared(np.asarray(np.random.normal(loc=0.0, scale= np.sqrt(1 / n_out), size=(n_in, n_out)),dtype=theano.config.floatX),name='w', borrow=True)
        self.b = theano.shared(np.asarray(np.random.normal(loc=0.0, scale=1.0, size=(n_out,)), dtype=theano.config.floatX), name='b', borrow=True)

        if (zerovs):
            self.w = theano.shared(np.zeros((n_in, n_out), dtype=theano.config.floatX), name='w', borrow=True)
            self.b = theano.shared(np.zeros((n_out,), dtype=theano.config.floatX), name='w', borrow=True)
        self.params = [self.w, self.b]

    def set_inpt(self, inpt):
        self.inpt = inpt.reshape((self.mini_batch_size, self.n_in))
        self.output = self.activation_fn(T.dot(self.inpt, self.w) + self.b)
        self.y_out = T.argmax(self.output, axis=1)

    def accuracy(self, y):
        "Return the accuracy for the mini-batch."
        return T.mean(T.eq(y, self.y_out))