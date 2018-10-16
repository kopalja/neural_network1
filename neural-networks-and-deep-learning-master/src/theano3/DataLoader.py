import pickle
import gzip
import theano
import numpy as np

def vectorized_result(j):
    e = np.zeros(10)
    e[j] = 1.0
    return e
    
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