
import pickle
import gzip
import theano
import numpy as np
import theano.tensor as T

import matplotlib.pyplot as plt


from FullyConectedLayer import *
from ConvPoolLayer import *
from DataLoader import *
from CostFunctions import *


def ReLU(z): return T.maximum(0.0, z)


def size(data):
    "Return the size of the dataset `data`."
    return data[0].get_value(borrow=True).shape[0]

def vectorized_result(j):
    e = np.zeros(10)
    e[j] = 1.0
    return e



class Net(object):

    def __init__(self, layers, training_data, validation_data, cost_fn, learning_rate, minibatch_size, regulation_param):
        training_x, training_y = training_data
        validation_x, validation_y = validation_data
        self.number_of_training_batches = size(training_data) // minibatch_size
        self.number_of_validation_batches = size(validation_data) // minibatch_size

        self.set_parameters(layers)
        self.build_computation_graph(layers, cost_fn)
        self.define_update(learning_rate)

        i = T.lscalar('index')
        self.train_by_minibatch = theano.function(
            [i],
            self.cost,
            updates = self.update,
            givens= {self.x: training_x[i * minibatch_size: (i+1) * minibatch_size], self.y: training_y[i * minibatch_size: (i+1) * minibatch_size]}
        )
        self.validate = theano.function(
            [i], 
            self.ac, 
            givens= {self.x: validation_x[i * minibatch_size: (i+1) * minibatch_size], self.y: validation_y[i * minibatch_size: (i+1) * minibatch_size]}
        )

    def set_parameters(self, layers):
        self.parameters = [layer.w for layer in layers]
        self.parameters += [layer.b for layer in layers]
      
    def build_computation_graph(self, layers, cost_fn):
        self.x = T.matrix('input')
        self.y = T.matrix('output')
        information = self.x
        for layer in layers:
            information = layer.feed_forward(information)
        self.cost = cost_fn(self.y, information)
        self.ac = self.accuraci(self.y, information)
 
    def define_update(self, learning_rate):
        grads = T.grad(self.cost, self.parameters)
        self.update = [(param, param - (learning_rate * grad)) for param, grad in zip(self.parameters, grads)]  

    def accuraci(self, x, y):
        results_ex, _ = theano.scan(fn = lambda vector: T.argmax(vector), sequences = x)
        results_obt, _ = theano.scan(fn = lambda vector: T.argmax(vector), sequences = y)
        return T.mean(T.eq(results_ex, results_obt))    

    def train(self, epoch):
        best_validation = 0
        for i in range(epoch):
            for index in range(self.number_of_training_batches):
                self.train_by_minibatch(index)
            current_validation = np.mean( [ self.validate(i) for i in range(self.number_of_validation_batches) ] )

            print("Epoch {0}: validation accuracy {1:.2%}".format(i, current_validation))


theano.config.floatX = 'float32'
training_data, validation_data, test_data = load_data_shared()
minibatch_size = 10

net = Net(
    layers = [
        #ConvPoolLayer(input_shape = (minibatch_size, 1, 28, 28), output_images = 10, kernel_size = 5, pool_shape=(2, 2), activation_fn = ReLU),
        ConvPoolLayer(input_shape = (minibatch_size, 1, 28, 28), output_images = 10, kernel_size = 5, pool_shape = (2, 2), activation_fn = ReLU),
        ConvLayer(input_shape = (minibatch_size, 10, 12, 12), output_images = 20, kernel_size = 5, activation_fn = ReLU),
        FullyConectedLayer((minibatch_size, 20 * 8 * 8), 30, activation_fn = T.nnet.sigmoid),
        FullyConectedLayer((minibatch_size, 30), 10, activation_fn = T.nnet.sigmoid)
    ], 
    training_data = training_data, 
    validation_data = validation_data, 
    cost_fn = CostFunctions().quadratic,
    learning_rate = 0.5, 
    minibatch_size = minibatch_size, 
    regulation_param = 0.05
)


net.train(epoch = 10)


