

import mnist_loader


import random

import numpy as np


class QuadraticCost(object):
    
    @staticmethod
    def fn(a, y):
        return 0.5*np.linalg.norm(a-y)**2

    @staticmethod
    def delta(z, a, y):
        return (a-y) * sigmoid_prime(z)


class CrossEntropyCost(object):

    @staticmethod
    def fn(a, y):
        return np.sum(np.nan_to_num(-y*np.log(a)-(1-y)*np.log(1-a)))

    @staticmethod
    def delta(z, a, y):
        return (a-y)


class MyNet(object):

    # Random init
    def __init__(self, sizes, cost_function):
        self.number_of_layers = len(sizes)
        self.sizes = sizes
        self.biases = [np.random.randn(item, 1)  for item in sizes[1:]]
        self.weights = [np.random.randn(sizes[item], sizes[item - 1])/np.sqrt(sizes[item - 1])  for item in range(1, len(sizes))]
        self.cost_function = cost_function
        
    
    # set hyper params
    def set_hyper_parameters(self, step_size, epoch, mini_batch_size, regulation):
        self.step_size = step_size
        self.epoch = epoch
        self.mini_batch_size = mini_batch_size
        self.regulation = regulation


    # train network
    def train(self, train_data, test_data):
        random.shuffle(train_data)
        #train_data = train_data[:1000]
        for i in range(self.epoch):
            random.shuffle(train_data)
            mini_batches = [train_data[j:j + self.mini_batch_size] for j in range(0, len(train_data), self.mini_batch_size)]
            
            for mini_batch in mini_batches:
                self.train_by_mini_batch(mini_batch, len(train_data))

            print("Epoch - test_data {0}: {1} / {2}".format(i, self.evaluate(test_data), len(test_data)))    
            #print("Epoch - train_data {0}: {1} / {2}".format(i, self.evaluate(train_data), len(train_data)))   
                
                
    def train_by_mini_batch(self, mini_batch, n):
        gradient_sum_b = [np.zeros(i.shape) for i in self.biases]
        gradient_sum_w = [np.zeros(i.shape) for i in self.weights]

        for input_record, output_record in mini_batch:
            gradient_b, gradient_w = self.back_propagation(input_record, output_record)
            gradient_sum_b = [x + y for x, y in zip(gradient_sum_b, gradient_b)]
            gradient_sum_w = [x + y for x, y in zip(gradient_sum_w, gradient_w)]

        self.biases = [old_b - (self.step_size / self.mini_batch_size) * gradient_b for old_b, gradient_b in zip(self.biases, gradient_sum_b)]
        self.weights = [(1.0 - self.step_size * (self.regulation / n)) * old_w - (self.step_size / self.mini_batch_size) * gradient_w for old_w, gradient_w in zip(self.weights, gradient_sum_w)]

            
    def back_propagation(self, input_record, output_record):
        gradient_b = [np.zeros(i.shape) for i in self.biases]
        gradient_w = [np.zeros(i.shape) for i in self.weights]
        z = []
        activations = [input_record]
        #forward
        for b, w in zip(self.biases, self.weights):
            z.append(np.dot(w, activations[-1]) + b)
            activations.append(self.sigmoid(z[-1]))
            
        # equation 1, 3: 
        gradient_b[-1] = self.cost_function.delta(z[-1], activations[-1], output_record)
        # equation 4: 
        gradient_w[-1] = np.dot(gradient_b[-1], activations[-2].transpose()) 

        for i in range(2, self.number_of_layers):
            # equation 2, 3: 
            gradient_b[-i] = np.dot(self.weights[-i + 1].transpose(), gradient_b[-i + 1]) * self.sigmoid_prime(z[-i])
            # equation 4: 
            gradient_w[-i] = np.dot(gradient_b[-i], activations[-i - 1].transpose())
        return (gradient_b, gradient_w)



    def evaluate(self, test_data):
        test_results = [(np.argmax(self.feedforward(x)), y) for (x, y) in test_data]
        return sum(int(x == y) for (x, y) in test_results)

    def feedforward(self, a):
        for b, w in zip(self.biases, self.weights):
            a = self.sigmoid(np.dot(w, a)+b)
        return a

    def sigmoid(self, x):
        return 1.0 / (1.0 + np.exp(-x))

    def sigmoid_prime(self, x):
        return self.sigmoid(x) * (1 - self.sigmoid(x)) 


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

def sigmoid_prime(x):
    return sigmoid(x) * (1 - sigmoid(x)) 









training_data, validation_data, test_data = mnist_loader.load_data_wrapper()

net = MyNet([784, 100, 10], CrossEntropyCost)

net.set_hyper_parameters(0.1, 30, 10, 5.0)

net.train(list(training_data), list(test_data))


