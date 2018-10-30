import pickle
import gzip

import theano
import numpy as np
import theano.tensor as T


def vectorized_result(j):
    e = np.zeros(10)
    e[j] = 1.0
    return e

#### Load the MNIST data
def load_data_shared(filename="../data/mnist.pkl.gz"):
    f = gzip.open(filename, 'rb')
    training_data, validation_data, test_data = pickle.load(f, encoding='latin1')

    # training_x, training_y = training_data
    # training_data = (training_x[:2000], training_y[:2000])

    f.close()
    def shared(data):

        v = [vectorized_result(sample) for sample in data[1]]

        shared_x = theano.shared(np.asarray(data[0], dtype=theano.config.floatX), borrow=True)
        shared_y = theano.shared(np.asarray(data[1], dtype=theano.config.floatX), borrow=True)
        #return shared_x, shared_y
        return shared_x, T.cast(shared_y, "int32")
    return [shared(training_data), shared(validation_data), shared(test_data)]