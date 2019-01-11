from abc import ABC, abstractmethod

import theano.tensor as T



class Layer():

    # Default layer properties.
    dropout = 0.0
    minibatch_size = 10

    def __init__(self, params : list):
        last_index = len(params) - 1
        activation_fn = params[last_index]
        if activation_fn == T.nnet.sigmoid:
            params[last_index] = 'sigmoid'
        elif activation_fn == T.nnet.softmax:
            params[last_index] = 'softmax'
        elif activation_fn == T.nnet.relu:
            params[last_index] = 'relu'
        else:
            self.ctor_params = params
            return
        self.ctor_params = params





    