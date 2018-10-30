"""network3.py
~~~~~~~~~~~~~~

A Theano-based program for training and running simple neural
networks.

Supports several layer types (fully connected, convolutional, max
pooling, softmax), and activation functions (sigmoid, tanh, and
rectified linear units, with more easily added).

When run on a CPU, this program is much faster than network.py and
network2.py.  However, unlike network.py and network2.py it can also
be run on a GPU, which makes it faster still.

Because the code is based on Theano, the code is different in many
ways from network.py and network2.py.  However, where possible I have
tried to maintain consistency with the earlier programs.  In
particular, the API is similar to network2.py.  Note that I have
focused on making the code simple, easily readable, and easily
modifiable.  It is not optimized, and omits many desirable features.

This program incorporates ideas from the Theano documentation on
convolutional neural nets (notably,
http://deeplearning.net/tutorial/lenet.html ), from Misha Denil's
implementation of dropout (https://github.com/mdenil/dropout ), and
from Chris Olah (http://colah.github.io ).

Written for Theano 0.6 and 0.7, needs some changes for more recent
versions of Theano.

"""
import time

from DataLoader import *
from FullyconectedLayer import *
from ConvLayer import *


# Third-party libraries
import numpy as np
import theano
import theano.tensor as T
# origin from theano.tensor.nnet import conv
from theano.tensor.nnet import softmax
from theano.tensor import shared_randomstreams
from theano.tensor.signal import pool

# Activation functions for neurons
def linear(z): return z
def ReLU(z): return T.maximum(0.0, z)
from theano.tensor.nnet import sigmoid

from CostFunctions import CostFunctions


import math

#### Constants
GPU = True
if GPU:
    print ("Trying to run under a GPU.  If this is not desired, then modify")
    #"network3.py\nto set the GPU flag to False."
    try: theano.config.device = 'gpu'
    except: pass # it's already set
    theano.config.floatX = 'float32'
else:
    print("Running with a CPU.  If this is not desired, then the modify ")
        #"network3.py to set\nthe GPU flag to True."  


#### Main class used to construct and train networks
class Network(object):

    def __init__(self, layers, mini_batch_size, cost_fn):
        """Takes a list of `layers`, describing the network architecture, and
        a value for the `mini_batch_size` to be used during training
        by stochastic gradient descent.

        """
        self.cost_fn = cost_fn
        self.layers = layers
        self.mini_batch_size = mini_batch_size
        self.params = [param for layer in self.layers for param in layer.params]
        self.x = T.matrix("x")
        self.y = T.ivector("y")
        #self.y = T.matrix("y")
        init_layer = self.layers[0]
        init_layer.set_inpt(self.x)
        for j in range(1, len(self.layers)):
            prev_layer, layer  = self.layers[j-1], self.layers[j]
            layer.set_inpt(prev_layer.output)
        self.output = self.layers[-1].output

    def SGD(self, training_data, epochs, mini_batch_size, eta, validation_data, test_data, lmbda=0.0):
        training_x, training_y = training_data
        validation_x, validation_y = validation_data
        test_x, test_y = test_data

        # compute number of minibatches for training, validetaation and testing
        num_training_batches =  size(training_data) // mini_batch_size
        num_validation_batches = size(validation_data) // mini_batch_size
        num_test_batches = size(test_data) // mini_batch_size

        # define the (regularized) cost function, symbolic gradients, and updates
        learning_rate = T.scalar(dtype=self.params[0].dtype)
        i = T.lscalar()

        l2_norm_squared = sum([(layer.w).sum() for layer in self.layers])
        #cost = self.layers[-1].cost(self.y) + 0.5*lmbda*l2_norm_squared/num_training_batches
        #cost = self.cost_fn(self.output, self.y)# + 0.5*lmbda*l2_norm_squared/num_training_batches
        cost = self.cost_r()
        grads = T.grad(cost, self.params)
        updates = [(param, param - eta * grad *  learning_rate) for param, grad in zip(self.params, grads)]

        # define functions to train a mini-batch, and to compute the
        # accuracy in validation and test mini-batches.

        train_mb = theano.function(
            [i, learning_rate], l2_norm_squared, updates= updates,
            givens={
                self.x:
                training_x[i*self.mini_batch_size: (i+1)*self.mini_batch_size],
                self.y:
                training_y[i*self.mini_batch_size: (i+1)*self.mini_batch_size]
            })
        validate_mb_accuracy = theano.function(
            #[i], self.accuracy(self.output, self.y),
            [i], self.accuracy_r(),
            givens={
                self.x:
                validation_x[i*self.mini_batch_size: (i+1)*self.mini_batch_size],
                self.y:
                validation_y[i*self.mini_batch_size: (i+1)*self.mini_batch_size]
            })

            
        # Do the actual training
        best_validation_accuracy = 0.0
        epoch = 0
        costr = 0
        while True:
            if (epoch > 5):
                return
            epoch += 1
            start = time.time()
            for minibatch_index in range(num_training_batches):
                iteration = num_training_batches*epoch+minibatch_index
                costr = train_mb(minibatch_index, self.learning_rate_function(epoch))
                if (math.isnan(costr)):
                    print("cost is naaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaan!!!!!!")
                if (minibatch_index + 1) == num_training_batches:
                    validation_accuracy = np.mean([validate_mb_accuracy(j) for j in range(num_validation_batches)])
                    print("Epoch {0}: validation accuracy {1:.2%}".format(epoch, validation_accuracy))
                    if validation_accuracy >= best_validation_accuracy:
                        print("         This is the best validation accuracy to date.")
                        best_validation_accuracy = validation_accuracy
                        best_iteration = iteration
                        # if test_data:
                        #     test_accuracy = np.mean([test_mb_accuracy(j) for j in range(num_test_batches)])
                        #     print('The corresponding test accuracy is {0:.2%}'.format(test_accuracy))
            end = time.time()
            end = end - start
            print("time for one epoch : ", end)



        print("Finished training network.")
        print("Best validation accuracy of {0:.2%} obtained at iteration {1}".format(best_validation_accuracy, best_iteration))
        print("Corresponding test accuracy of {0:.2%}".format(test_accuracy))

    def learning_rate_function(self, epoch):
        return 1.0 / epoch
        
    def accuracy(self, output_batch, desire_output_batch):
        output_results, _ = theano.scan(fn = lambda output_vector: T.argmax(output_vector), sequences = output_batch)
        desire_results, _ = theano.scan(fn = lambda desire_output_vector: T.argmax(desire_output_vector), sequences = desire_output_batch)
        return T.mean(T.eq(output_results, desire_results))  
    
    def cost_r(self):
        return -1 * T.mean(T.log(self.output)[T.arange(self.y.shape[0]), self.y])

    def accuracy_r(self):
        "Return the accuracy for the mini-batch."
        return T.mean(T.eq(self.y, self.layers[-1].y_out))

# class SoftmaxLayer(object):

#     def __init__(self, mini_batch_size, n_in, n_out):
#         self.mini_batch_size = mini_batch_size
#         self.n_in = n_in
#         self.n_out = n_out
#         # Initialize weights and biases
#         self.w = theano.shared(
#             np.zeros((n_in, n_out), dtype=theano.config.floatX),
#             name='w', borrow=True)
#         self.b = theano.shared(
#             np.zeros((n_out,), dtype=theano.config.floatX),
#             name='b', borrow=True)
#         self.params = [self.w, self.b]

#     def set_inpt(self, inpt):
#         self.inpt = inpt.reshape((self.mini_batch_size, self.n_in))
#         self.output = softmax(T.dot(self.inpt, self.w) + self.b)
#         self.y_out = T.argmax(self.output, axis=1)


#     def cost(self, y):
#         expected_results, _ = theano.scan(
#             fn = lambda desire_output_vector: T.argmax(desire_output_vector), 
#             sequences = y
#         )
#         cost, _ = theano.scan(
#             fn = lambda output_vector, expected_result: T.log(output_vector[expected_result]),
#             sequences = [self.output, expected_results] 
#         )
#         return - T.mean(cost)


#     def cost_origin(self, y):
#         "Return the log-likelihood cost."
#         return -1 * T.mean(T.log(self.output)[T.arange(y.shape[0]), y])


#     def accuracy(self, y):
#         output_results, _ = theano.scan(fn = lambda output_vector: T.argmax(output_vector), sequences = self.output)
#         desire_results, _ = theano.scan(fn = lambda desire_output_vector: T.argmax(desire_output_vector), sequences = y)
#         return T.mean(T.eq(output_results, desire_results))   

#     def accuracy_origin(self, y):
#         "Return the accuracy for the mini-batch."
#         return T.mean(T.eq(y, self.y_out))


#### Miscellanea
def size(data):
    "Return the size of the dataset `data`."
    return data[0].get_value(borrow=True).shape[0]



training_data, validation_data, test_data = load_data_shared()
mini_batch_size = 10

expanded_data, _, _ = load_data_shared("../data/mnist_expanded.pkl.gz")


print("test1")

for i in range(15):
    net = Network([
        ConvPoolLayer(image_shape=(mini_batch_size, 1, 28, 28),
                        filter_shape=(5, 1, 5, 5), 
                        poolsize=(2, 2),
                        activation_fn=ReLU),
        ConvPoolLayer(image_shape=(mini_batch_size, 5, 12, 12),
                        filter_shape=(10, 5, 5, 5),
                        poolsize=(2, 2),
                        activation_fn=ReLU),
        FullyConnectedLayer(mini_batch_size, n_in=10*4*4, 
                        n_out=30,
                        activation_fn=ReLU),
        FullyConnectedLayer(mini_batch_size, n_in=30, n_out=10, activation_fn = T.nnet.softmax, zerovs = True)], mini_batch_size, CostFunctions.probability)
          
    net.SGD(training_data, 60, mini_batch_size, 0.03, validation_data, test_data, lmbda=0)


