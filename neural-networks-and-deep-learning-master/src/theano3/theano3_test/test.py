import sys
sys.path.append('C:\\xps\\Python\\Neural-Network\\neural-networks-and-deep-learning-master\\src\\theano3')

import collections
import unittest
import Layer
import FullyConectedLayer
from FullyConectedLayer import FullyConectedLayer
import numpy as np
import theano
import theano.tensor as T

class Test(unittest.TestCase):
    


    def test_feed_forward(self):
        inpt = T.vector()
        minibatch_size = T.iscalar()
        net = FullyConectedLayer(in_size = 2, out_size = 2, activation_fn = T.nnet.sigmoid)
        Layer.Layer.minibatch_size = minibatch_size
        o1 = net.feed_forward(inpt)
        o2 = T.nnet.sigmoid(T.dot(inpt.reshape((minibatch_size, 2)), net.w) + net.b)
        test = theano.function(inputs = [inpt, minibatch_size], outputs = [o1, o2])


        for minibatch_size in range(1, 5):
            inpt = np.asarray(np.random.normal(0.0, 1.0, minibatch_size * 2), dtype='float32')
            o1, o2 = test(inpt, minibatch_size)
            #self.assertEqual(collections.Counter(o1) == collections.Counter(o2))
            for x, y in zip(o1, o2):
                self.assertEqual(x[0], y[0])
                self.assertEqual(x[1], y[1])


if __name__ == '__main__':
    unittest.main()