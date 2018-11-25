# import sys
# sys.path.append('C:\\xps\\Python\\Neural-Network\\neural-networks-and-deep-learning-master\\src\\theano3')


import unittest
import numpy
import theano

class TestTensorDot(unittest.TestCase):
    def setUp(self):
        # data which will be used in various test methods
        self.avals = numpy.array([[1,5,3],[2,4,1]])
        self.bvals = numpy.array([[2,3,1,8],[4,2,1,1],[1,4,8,5]])
    

    def test_validity(self):
        a = theano.tensor.dmatrix('a')
        b = theano.tensor.dmatrix('b')
        c = theano.tensor.dot(a, b)
        f = theano.function([a, b], [c])
        cmp = f(self.avals, self.bvals) == numpy.dot(self.avals, self.bvals)
        self.assertTrue(numpy.all(cmp))


if __name__ == '__main__':
    unittest.main()