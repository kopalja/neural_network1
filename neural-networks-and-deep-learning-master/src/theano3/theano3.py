####Imports##########################################
import theano
import theano.tensor as T
import numpy as np
import GlobalProperties
from FullyConectedLayer import *
from ConvLayer import *
from DataLoader import *
from PoolLayer import *
from CostFunctions import CostFunctions as Cf
####Constants##########################################
theano.config.floatX = 'float32'
theano.compile.mode.Mode(optimizer = 'fast_compile')
########################################################
class Net:
    
    def __init__(self, layers, training_data, validation_data, cost_fn, learning_rate, minibatch_size, regulation_param):
        train_x, train_y = training_data
        valid_x, valid_y = validation_data
        self.number_of_training_batches = training_data[0].get_value(True).shape[0] // minibatch_size
        self.number_of_validation_batches = validation_data[0].get_value(True).shape[0] // minibatch_size

        self.set_parameters(layers)
        self.build_computation_graph(layers, cost_fn, regulation_param)
        self.define_update(learning_rate)
        
        i = T.lscalar('index')
        self.train_by_minibatch = theano.function(
            [i],
            self.cost,
            updates = self.update,
            givens= {
                self.input_batch: train_x[i * minibatch_size : (i+1) * minibatch_size], 
                self.desire_output_batch: train_y[i * minibatch_size : (i+1) * minibatch_size] }
        )
        self.validate = theano.function(
            [i], 
            self.correct_clasified, 
            givens= {
                self.input_batch: valid_x[i * minibatch_size : (i+1) * minibatch_size], 
                self.desire_output_batch: valid_y[i * minibatch_size : (i+1) * minibatch_size] }
        )

    def set_parameters(self, layers):
        self.parameters = [layer.w for layer in layers if layer.w != []]
        self.parameters += [layer.b for layer in layers  if layer.b != []]
      
    def build_computation_graph(self, layers, cost_fn, regulation_param):
        self.input_batch = T.matrix('input')
        self.desire_output_batch = T.matrix('output')
        batch = self.input_batch
        for layer in layers:
            batch = layer.feed_forward(batch)

        #l2_norm_squared = sum([T.sum(l.w ** 2) for l in layers if l.w != []])
        self.cost = cost_fn(batch, self.desire_output_batch) + self.regulation(layers, regulation_param) #regulation_param * (0.5 * l2_norm_squared / self.number_of_training_batches)   
        self.correct_clasified = self.batch_accuraci(batch, self.desire_output_batch)
        

 
    def define_update(self, learning_rate):
        grads = T.grad(self.cost, self.parameters)
        self.update = [(param, param - learning_rate * grad) for param, grad in zip(self.parameters, grads)]  
    
    def regulation(self, layers, regulation_param):
        l2_norm_squared = sum([T.sum(l.w ** 2) for l in layers if l.w != []])
        return regulation_param * (0.5 * l2_norm_squared / self.number_of_training_batches)

    def batch_accuraci(self, output_batch, desire_output_batch):
        output_results = T.argmax(output_batch, axis=1)
        desire_results = T.argmax(desire_output_batch, axis=1)
        return T.mean(T.eq(output_results, desire_results))    

    def train(self, epoch):
        for i in range(epoch):
            for index in range(self.number_of_training_batches):
                self.train_by_minibatch(index)
            current_validation = np.mean( [ self.validate(i) for i in range(self.number_of_validation_batches) ] )
            print("Epoch {0}: {1:.2%}".format(i, current_validation))
#######################################################






training_data, validation_data, test_data = load_data_shared()
#expanded_data, _, _ = load_data_shared("../data/mnist_expanded.pkl.gz")

for i in range(20):
    net = Net(
        layers = (
            ConvLayer(input_shape = (1, 28, 28), output_images = 10, kernel_size = 5, activation_fn = T.nnet.relu),
            PoolLayer(size = (2, 2)),
            ConvLayer(input_shape = (10, 12, 12), output_images = 20, kernel_size = 5, activation_fn = T.nnet.relu),
            PoolLayer(size = (2, 2)),
            FullyConectedLayer(in_size = 20 * 4 * 4, out_size = 200, activation_fn = T.nnet.relu),
            FullyConectedLayer(in_size = 200, out_size = 10, activation_fn = T.nnet.softmax)
        ), 
        training_data = training_data, 
        validation_data = validation_data, 
        cost_fn = Cf.probability,
        learning_rate = 0.03, 
        minibatch_size = GlobalProperties.mini_batch_size, 
        regulation_param = 0.0
    )
    print("now")
    net.train(epoch = 15)


