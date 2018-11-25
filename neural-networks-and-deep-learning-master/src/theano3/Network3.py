####Imports##########################################
import numpy as np
import theano
import theano.tensor as T
from Layer import Layer
from ConvLayer import ConvLayer
from PoolLayer import PoolLayer
from FullyConectedLayer import FullyConectedLayer
from BatchNormalizationLayer import BatchNormalizationLayer
from CostFunctions import CostFunctions as Cf
from DnnLoader import DnnLoader
from abc import ABC, abstractmethod
import time
####Temp###############################################
import matplotlib.pyplot as plt
####Enum################################################
from enum import Enum
class Update(Enum):
    Vanilla = 1
    Momentum = 2
    Adagrad = 3
####Constants###########################################
theano.config.floatX = 'float32'
theano.compile.mode.Mode(optimizer = 'fast_run')
########################################################
class Network:
    def __init__(self, layers, cost_fn, learning_rate, minibatch_size, dropout, regulation_param, update_type, load_net_file):
        if (load_net_file != 'none'):
            layers = DnnLoader.Load(load_net_file)
        Layer.dropout = dropout
        Layer.minibatch_size = minibatch_size   
 
        self.__set_parameters(layers, update_type)
        self.__build_computation_graph(layers, cost_fn, regulation_param)
        self.__define_update(learning_rate, update_type)
        self.layers = layers
            
    def __set_parameters(self, layers, update_type):
        self.parameters = [param for layer in layers for param in layer.params]
        if update_type == Update.Momentum or update_type == Update.Adagrad:
            self.cache = [0.0 for layer in layers for param in layer.params]

    def __build_computation_graph(self, layers, cost_fn, regulation_param):
        self.input_batch = T.matrix('input')
        self.desire_output_batch = T.matrix('output')
        train_d =  self.input_batch
        valid = self.input_batch
        for layer in layers:
            train_d = layer.feed_forward_dropout(train_d)
            valid = layer.feed_forward(valid)
        self.cost = cost_fn(train_d, self.desire_output_batch) + self.__regulation(layers, regulation_param)
        # definied by child   
        self.validation = self.result(valid, self.desire_output_batch)
        
    def __define_update(self, learning_rate, update_type):
        grads = T.grad(self.cost, self.parameters)
        if update_type == Update.Momentum:
            print("use momentum")
            self.cache = [0.9 * cache - learning_rate * grad  for cache, grad in zip(self.cache, grads)]
            self.update = [(param, param + cache) for param, cache in zip(self.parameters, self.cache)]
        elif update_type == Update.Adagrad:
            print("use adgrad")
            self.cache = [cache + T.pow(grad, 2) for cache, grad in zip(self.cache, grads)]
            self.update = [(param, param - learning_rate * grad / T.sqrt(cache)) for param, grad, cache in zip(self.parameters, grads, self.cache)]
        else:
            print("use sgd")
            self.update = [(param, param - learning_rate * grad) for param, grad in zip(self.parameters, grads)]      

    def __regulation(self, layers, regulation_param):
        l2_norm_squared = sum([T.sum(l.w ** 2) for l in layers if l.w != []])
        return regulation_param * (0.5 * l2_norm_squared / 1000000)
    
    @abstractmethod
    def result(self, output, desired_output):
        NotImplementedError("result on Network was called")


#######################################################
class Batch_Network(Network):
    def __init__(self, layers, training_data, validation_data, cost_fn, learning_rate, minibatch_size, dropout, regulation_param, update_type, normalize_data, load_from_file = 'none', save_to_file = 'none'):
        Network.__init__(self, layers, cost_fn, learning_rate, minibatch_size, dropout, regulation_param, update_type, load_from_file)

        # Prepare data. Load data to shared memory. Data should be float32 dtype
        train_in, train_out = self.__convert_and_load(training_data, normalize_data)
        valid_in, valid_out = self.__convert_and_load(validation_data, normalize_data)

        i = T.lscalar('index')
        self.__train_by_minibatch = theano.function(inputs = [i], outputs = self.cost, updates = self.update,
            givens= {
                self.input_batch: train_in[i * minibatch_size : (i+1) * minibatch_size], 
                self.desire_output_batch: train_out[i * minibatch_size : (i+1) * minibatch_size] }
        )
        self.__validate = theano.function(inputs = [i], outputs = self.validation, 
            givens= {
                self.input_batch: valid_in[i * minibatch_size : (i+1) * minibatch_size], 
                self.desire_output_batch: valid_out[i * minibatch_size : (i+1) * minibatch_size] }
        )   
        self.save_to_file = save_to_file
        self.number_of_training_batches = len(training_data[0]) // minibatch_size
        self.number_of_validation_batches = len(validation_data[0]) // minibatch_size 
        return
    
    
    def __convert_and_load(self, data, norm):
        """ Convert data to float32 and load to shared memory for GPU usage """
        x = data[0]
        if norm:
            mean = np.mean(x, axis = 1)
            std = np.std(x, axis = 1) 
            x = (x - np.expand_dims(mean, 1)) / np.expand_dims(std, 1)
        x = theano.shared(np.asarray(x, dtype = theano.config.floatX), borrow = True)
        y = theano.shared(np.asarray(data[1], dtype = theano.config.floatX), borrow = True)
        return x, y

    def result(self, output_batch, desire_output_batch):
        output_results = T.argmax(output_batch, axis = 1)
        desire_results = T.argmax(desire_output_batch, axis = 1)
        return T.mean(T.eq(output_results, desire_results))  

    def train(self, epoch):
        r = 0
        for i in range(epoch):
            start = time.time()
            cost = np.zeros(self.number_of_training_batches)
            for index in range(self.number_of_training_batches):
                cost[index] = self.__train_by_minibatch(index)
            end = time.time()
            print("epoch time ", end - start)
            current_validation = np.mean( [ self.__validate(i) for i in range(self.number_of_validation_batches) ] )
            print("Epoch {0}: {1:.2%}".format(i, current_validation))
        DnnLoader.Save(self.save_to_file, self.layers)
#######################################################
class Online_NetWork(Network):
    def __init__(self, layers, cost_fn, learning_rate, dropout, regulation_param, momentum = False):
        Network.__init__(self, layers, cost_fn, learning_rate, 1, dropout, regulation_param, momentum)

        input_vector = T.vector('input_vector')
        desire_output_vector = T.vector('desire_output_vector')
        self.train_online = theano.function(inputs = [input_vector, desire_output_vector], outputs = self.cost, updates = self.update,
            givens = {
                self.input_batch: input_vector.reshape((1, input_vector.shape[0])), 
                self.desire_output_batch: desire_output_vector.reshape((1, desire_output_vector.shape[0]))}
        )
        self.validate_online = theano.function([input_vector], self.validation,
            givens = {self.input_batch: input_vector.reshape((1, input_vector.shape[0]))}
        )

    def result(self, output_batch, desire_output_batch):
        return T.argmax(output_batch, axis = 1)
#######################################################

    
