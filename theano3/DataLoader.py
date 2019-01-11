import pickle
import gzip
#import theano
import numpy as np


import cv2
from os import listdir
from os.path import isfile, join
from PIL import Image

def vectorized_result(j):
    e = np.zeros(10)
    e[j] = 1.0
    return e
    
def load_data(filename="data/mnist.pkl.gz"):
    f = gzip.open(filename, 'rb')
    training_data, validation_data, test_data = pickle.load(f, encoding='latin1')

    f.close()
    def vector(data):
        vector_r = [vectorized_result(sample) for sample in data[1]]
        return data[0], vector_r
    return [vector(training_data), vector(validation_data), vector(test_data)]

