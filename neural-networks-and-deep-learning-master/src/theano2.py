
import pickle
import gzip
import theano
import numpy as np
import theano.tensor as T

import matplotlib.pyplot as plt



class CostFunctions(object):

    def quadratic(self, output_batch, desire_output_batch):
        def vector_distance(output_vector, desire_output_vector):
            distance, _ = theano.scan(lambda x, y: (x - y) ** 2, [output_vector, desire_output_vector])
            return T.sum(distance)

        cost, _ = theano.scan(
            fn = lambda x, y: vector_distance(x, y),
            sequences = [output_batch, desire_output_batch]
        )
        return T.mean(cost) * 0.5

    def crossEntropy(self, output_batch, desire_output_batch):
        def vector_entropy_distance(output_vector, desire_output_vector):
            distance, _ = theano.scan(lambda x, y: y * T.log(x) + (1.0 - y) * T.log(1.0 - x), [output_vector, desire_output_vector])
            return distance

        distances, _ = theano.scan(
            fn = lambda output_vector, desire_output_vector: vector_entropy_distance(output_vector, desire_output_vector),
            sequences = [output_batch, desire_output_batch]
        )
        return - T.mean(distances)

    def probability(self, output_batch, desire_output_batch):
        expected_results, _ = theano.scan(
            fn = lambda desire_output_vector: T.argmax(desire_output_vector), 
            sequences = desire_output_batch
        )
        cost, _ = theano.scan(
            fn = lambda output_vector, expected_result: T.log(output_vector[expected_result]),
            sequences = [output_batch, expected_results] 
        )
        return - T.mean(cost)









#### Load the MNIST data
def load_data_shared(filename="../data/mnist.pkl.gz"):
    f = gzip.open(filename, 'rb')
    training_data, validation_data, test_data = pickle.load(f, encoding='latin1')

    
    f.close()
    def shared(data):
        """Place the data into shared variables.  This allows Theano to copy
        the data to the GPU, if one is available.

        """
        vector_r = [vectorized_result(sample) for sample in data[1]]

        shared_x = theano.shared(np.asarray(data[0], dtype=theano.config.floatX))
        shared_y = theano.shared(np.asarray(vector_r, dtype=theano.config.floatX))
        return shared_x, shared_y
    return [shared(training_data), shared(validation_data), shared(test_data)]

def size(data):
    "Return the size of the dataset `data`."
    return data[0].get_value(borrow=True).shape[0]

def vectorized_result(j):
    e = np.zeros(10)
    e[j] = 1.0
    return e

class Layer(object):
    def __init__(self, in_size, out_size):
        self.b = theano.shared(np.asarray(np.random.normal(0, 1, out_size), dtype = theano.config.floatX))
        self.w = theano.shared(np.array(np.random.normal(0, np.sqrt(1.0 / out_size), (in_size, out_size )), dtype = theano.config.floatX))
        self.params = [self.b, self.w]
        self.in_size = in_size


class Net(object):

    def __init__(self, layers, train_data, validation_data1, validation_data2, costFunction, learning_rate, minibatch_size, regulation_param):
        
        self.training_x, self.training_y = train_data
        self.training_x = self.training_x
        self.training_y = self.training_y


        self.costFunction = costFunction
        self.learning_rate = learning_rate
        self.minibatch_size = minibatch_size
        self.number_of_training_batches = size(train_data) // minibatch_size
        self.number_of_validation_batches = size(validation_data1) // minibatch_size
        self.validation_x1, self.validation_y1 = validation_data1
        self.validation_x2, self.validation_y2 = validation_data2
        self.regulation_param = regulation_param


        # init layers
        self.l = []
        for i in range(len(layers) - 1):
            self.l.append(Layer(layers[i], layers[i + 1]))
  
        self.params = [layer.w for layer in self.l]
        self.params += [layer.b for layer in self.l]
        #self.params = [param for layer in self.l for param in layer.params]    
        self.speed = [0 for i in self.params]


    def sigmoid(self, x):
        return 1.0 / ( 1.0 + T.exp(-x) ) 


    def prepare_and_compile_function(self):
        x = T.matrix('x')
        y = T.matrix('y')
        epoch = T.scalar()
        index = T.lscalar('index')

        # symbolic forward function
        output = x
        for i in range(len(self.l)):
            output = self.sigmoid(T.dot(output, self.l[i].w) + self.l[i].b)

        #output = self.softmax(T.dot(output, self.l[-1].w) + self.l[-1].b)


        cost = self.costFunction(output, y)
        accuraci = self.accuraci(output, y)
        grads = T.grad(cost, self.params)
        self.speed = [ speed - self.learning_rate * grad  for speed, grad in zip(self.speed, grads)]
        #updates = [(param, param + speed ) for param, speed in zip(self.params, self.speed)]



        updates = [(param, param - self.learning_rate * grad ) for param, grad in zip(self.params, grads)]

        down_index = index * self.minibatch_size

        self.train_by_minibatch = theano.function(
            [index, epoch], 
            self.regulation(), 
            updates = updates, 
            givens = { x: self.training_x[down_index : down_index + self.minibatch_size], y: self.training_y[down_index:down_index + self.minibatch_size] },
            on_unused_input = 'ignore'
        )

        self.validate1 = theano.function(
            [index], 
            accuraci, 
            givens = { x: self.validation_x1[down_index : down_index + self.minibatch_size], y: self.validation_y1[down_index:down_index + self.minibatch_size] },
        )

        self.validate2 = theano.function(
            [index], 
            accuraci, 
            givens = { x: self.validation_x2[down_index : down_index + self.minibatch_size], y: self.validation_y2[down_index:down_index + self.minibatch_size] },
        )

    def accuraci(self, x, y):
        results_ex, _ = theano.scan(fn = lambda vector: T.argmax(vector), sequences = x)
        results_obt, _ = theano.scan(fn = lambda vector: T.argmax(vector), sequences = y)
        return T.mean(T.eq(results_ex, results_obt))  
    
    def regulation(self):
        summ = theano.shared(0)
        for layer in self.l:
            s, _ = theano.scan(
                fn = lambda vector: T.sum(vector ** 2),
                sequences = layer.w
            )
            #print(T.sum(s).eval())
            summ += T.sum(s)

        summ1 = sum([T.sum(l.w ** 2) for l in self.l])

        return summ, summ1

        return ( summ * self.regulation_param * 0.5 ) / ( self.number_of_training_batches * self.minibatch_size )

    def softmax(self, output_batch):
        sum_batch, _ = theano.scan(fn = lambda output_vector: T.sum(T.exp(output_vector)), sequences = output_batch)
        out, _ = theano.scan(fn = lambda output_vector, summ: T.exp(output_vector) / summ, sequences = [output_batch, sum_batch])
        return out


    def train(self, epoch):
        results = []
        best_validation = 0
        for i in range(epoch):
            for index in range(self.number_of_training_batches):
                s1, s2 = self.train_by_minibatch(index, i + 1)
                #print("sum1 ", s1)
                #print("sum2 ", s2)

            current_validation1 = np.mean( [ self.validate1(i) for i in range(self.number_of_validation_batches) ] )
            current_validation2 = np.mean( [ self.validate2(i) for i in range(self.number_of_validation_batches) ] )



            print("Epoch {0}: positive accuracy {1:.2%}".format(i, current_validation1))
            print("Epoch {0}: negative accuracy {1:.2%}".format(i, current_validation2))






# theano.config.floatX = 'float32'

# training_data, validation_data, test_data = load_data_shared()

# print(validation_data)



# net = Net(
#     [784, 30, 10], 
#     training_data, 
#     validation_data, 
#     CostFunctions().quadratic,
#     learning_rate = 1.5, 
#     minibatch_size = 10, 
#     regulation_param = 0.0
# )

# net.prepare_and_compile_function()

# net.train(epoch = 10)


