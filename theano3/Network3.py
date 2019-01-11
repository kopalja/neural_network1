####Imports##########################################
import theano
import theano.tensor as T
import numpy as np
from LayersWrapper import LayersWrapper
from Layer import Layer
from CostFunctions import CostFunctions as Cf
from DnnLoader import DnnLoader
from abc import ABC, abstractmethod
import time
####Temp###############################################
import matplotlib.pyplot as plt
####Enum################################################
from enum import Enum
class Update(Enum):
    Sgd = 1
    Momentum = 2
    Adagrad = 3
    RMSProp = 4
    Adam = 5
####Constants###########################################
theano.config.floatX = 'float32'
theano.compile.mode.Mode(optimizer = 'fast_run')
########################################################
class Network:
    def __init__(self, layers, cost_fn, learning_rate, minibatch_size, dropout, l2_regulation, update_type, load_net_file):
        self._trainning_func_output = 0
        self._star_time_stamp = time.time()
        self.__create_layers(layers, minibatch_size, dropout, load_net_file)
        self.__set_learning_parameters(update_type)
        cost_graph = self.__build_computation_graph(cost_fn, l2_regulation)
        self._network_update = self.__define_network_update(cost_graph, learning_rate, update_type)
    
    def __create_layers(self, layers, minibatch_size, dropout, load_net_file):
        """ Load or create network layers if needed. """
        if load_net_file != None:
            self._layers = DnnLoader.Load(load_net_file)
        elif type(layers) == LayersWrapper:
            self._layers = layers.create_layers()
        else:
            self._layers = layers
        Layer.minibatch_size = minibatch_size
        Layer.dropout = dropout

    def __set_learning_parameters(self, update_type):
        """ Wrap all learning parameters into one list. """
        self.__parameters = [param for layer in self._layers for param in layer.params]
        #if update_type != Update.Sgd:
        self.__cache1 = [theano.shared(np.zeros(param.shape.eval()).astype('float32')) for layer in self._layers for param in layer.params]
        if update_type == Update.Adagrad or update_type == Update.Adam or update_type == Update.RMSProp:
            self.__cache2 = [theano.shared(np.zeros(param.shape.eval()).astype('float32')) for layer in self._layers for param in layer.params]
            self.__epsylon = 0.01

    def __build_computation_graph(self, cost_fn, l2_regulation):
        """ Create computation graph to count network cost. """
        self._input_batch = T.matrix('input')
        self._desire_output_batch = T.matrix('output')
        train_feed_forward =  self._input_batch
        valid_feed_forward = self._input_batch
        for layer in self._layers:
            train_feed_forward = layer.feed_forward_dropout(train_feed_forward)
            valid_feed_forward = layer.feed_forward(valid_feed_forward)
        # definied by child   
        self._validation_func_output = self._result_interpretation(valid_feed_forward, self._desire_output_batch)
        self._trainning_func_output = cost_fn(train_feed_forward, self._desire_output_batch)
        return self._trainning_func_output
        
    def __define_network_update(self, cost_graph, learning_rate, update_type):
        """ Based on network parameters and cost define parameters update. """
        grads = T.grad(cost_graph, self.__parameters)
        self._trainning_func_output = grads
        if update_type == Update.Sgd:
            parameters_update = [(param, param - learning_rate * grad) for param, grad in zip(self.__parameters, grads)]  
            return parameters_update
        elif update_type == Update.Momentum:
            velocity_new_value = [0.9 * v_old - learning_rate * grad  for v_old, grad in zip(self.__cache1, grads)]
            velocity_update = [(v_old, v_new)  for v_old, v_new in zip(self.__cache1, velocity_new_value)]
            parameters_update = [(param, param + v_new) for param, v_new in zip(self.__parameters, velocity_new_value)]
            return velocity_update + parameters_update
        elif update_type == Update.Adagrad:
            cache1_update = [(cache, cache + T.pow(grad, 2)) for cache, grad in zip(self.__cache1, grads)]
            parameters_update = [(param, param - learning_rate * grad / (T.sqr(cache) + self.__epsylon)) for param, grad, cache in zip(self.__parameters, grads, self.__cache1)]
            return cache1_update + parameters_update
        elif update_type == Update.RMSProp:
            decay = 0.99
            cache1_update = [(cache, decay * cache + (1.0 - decay) * T.pow(grad, 2)) for cache, grad in zip(self.__cache1, grads)]
            parameters_update = [(param, param - learning_rate * grad / (T.sqr(cache) + self.__epsylon)) for param, grad, cache in zip(self.__parameters, grads, self.__cache1)]    
            return cache1_update + parameters_update
        elif update_type == Update.Adam:
            beta1 = 0.9
            beta2 = 0.99
            self.__timestamp = theano.shared(np.float32(1))
            timestamp_update = [(self.__timestamp, self.__timestamp + 1)]
            cache1_update = [(cache1, beta1 * cache1 + (1.0 - beta1) * grad) for cache1, grad in zip(self.__cache1, grads)]
            cache2_update = [(cache2, beta2 * cache2 + (1.0 - beta2) * T.pow(grad, 2)) for cache2, grad in zip(self.__cache2, grads)]
            parameters_update = [(param, param - learning_rate * (cache1 / (1.0 - T.pow(beta1, self.__timestamp)))  / (T.sqr(cache2 / (1.0 - T.pow(beta2, self.__timestamp))) + self.__epsylon)) for param, grad, cache1, cache2 in zip(self.__parameters, grads, self.__cache1, self.__cache2)] 
            return timestamp_update + cache1_update + cache2_update + parameters_update         

    @abstractmethod
    def _result_interpretation(self, output, desired_output):
        """ Each child define how to interpret network result. """
        NotImplementedError("result on Network was called")


#######################################################
class Batch_Network(Network):
    def __init__(self, layers, training_data, validation_data, cost_fn, learning_rate, minibatch_size, dropout, l2_regulation, update_type, normalize_data = False, load_from_file = None, save_to_file = None):
        Network.__init__(self, layers, cost_fn, learning_rate, minibatch_size, dropout, l2_regulation, update_type, load_from_file)
        # Prepare data. Load data to shared memory. Data should be float32 dtype
        train_in, train_out = self.__convert_and_load(training_data, normalize_data)
        valid_in, valid_out = self.__convert_and_load(validation_data, normalize_data)
        i = T.lscalar('index')
        self.__train_by_minibatch = theano.function(inputs = [i], outputs = self._trainning_func_output, updates = self._network_update,
            givens= { self._input_batch: train_in[i * minibatch_size : (i+1) * minibatch_size], 
            self._desire_output_batch: train_out[i * minibatch_size : (i+1) * minibatch_size] }
        )
        self.__validate = theano.function(inputs = [i], outputs = self._validation_func_output, 
            givens= { self._input_batch: valid_in[i * minibatch_size : (i+1) * minibatch_size], 
            self._desire_output_batch: valid_out[i * minibatch_size : (i+1) * minibatch_size] }
        )   
        self.__save_to_file = save_to_file
        self.__number_of_training_batches = len(training_data[0]) // minibatch_size
        self.__number_of_validation_batches = len(validation_data[0]) // minibatch_size 
        print("compilation time ", time.time() - self._star_time_stamp)
        print('==================================')

    def __convert_and_load(self, data, norm):
        """ Convert data to float32 and load to shared memory for GPU usage """
        temp = data[0]
        if norm:
            mean = np.mean(data[0], axis = 1)
            std = np.std(data[0], axis = 1) 
            temp = (data[0] - np.expand_dims(mean, 1)) / np.expand_dims(std, 1)
        x = theano.shared(np.asarray(temp, dtype = theano.config.floatX), borrow = True)
        y = theano.shared(np.asarray(data[1], dtype = theano.config.floatX), borrow = True)
        return x, y

    def _result_interpretation(self, output_batch, desire_output_batch):
        output_results = T.argmax(output_batch, axis = 1)
        desire_results = T.argmax(desire_output_batch, axis = 1)
        return T.mean(T.eq(output_results, desire_results))  


    def train(self, epoch):
        best_validation = 0
        for i in range(epoch):
            start = time.time()
            for index in range(self.__number_of_training_batches):
                out = self.__train_by_minibatch(index)
            print(np.max(out[0]))
            end = time.time()
            print("epoch time ", end - start)
        
            current_validation = np.mean( [ self.__validate(i) for i in range(self.__number_of_validation_batches) ] )
            print("Epoch {0}: {1:.2%}".format(i, current_validation))
            if (current_validation > best_validation):
                best_validation = current_validation
                DnnLoader.Save(self.__save_to_file, self._layers)
        # plt.plot(output)
        # plt.show()
            

    def test(self):
        current_validation = np.mean( [ self.__validate(i) for i in range(self.__number_of_validation_batches) ] )
        print("Validation accuracy: {0:.2%}".format(current_validation))
#######################################################
class Online_NetWork(Network):
    def __init__(self, layers, cost_fn, learning_rate, dropout, l2_regulation, momentum = False):
        Network.__init__(self, layers, cost_fn, learning_rate, 1, dropout, l2_regulation, momentum, load_net_file = None)

        input_vector = T.vector()
        desire_output_vector = T.vector('desire_output_vector')
        self.train_online = theano.function(inputs = [input_vector, desire_output_vector], outputs = None, updates = self._network_update,
            givens = {
                self._input_batch: input_vector.reshape((1, input_vector.shape[0])), 
                self._desire_output_batch: desire_output_vector.reshape((1, desire_output_vector.shape[0]))}
        )
        self.validate_online = theano.function([input_vector], self._validation_func_output,
            givens = {self._input_batch: input_vector.reshape((1, input_vector.shape[0]))}
        )

    def _result_interpretation(self, output_batch, desire_output_batch):
        return T.argmax(output_batch, axis = 1)
#######################################################

    
