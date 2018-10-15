

import theano 
import numpy as np
import theano.tensor as T


w1 = theano.shared(np.array(np.random.normal(0, 1, (5, 5 )), dtype = theano.config.floatX))

w2 = theano.shared(np.array(np.random.normal(0, 1, (5, 5 )), dtype = theano.config.floatX))


s = T.sum(w1)
s2 = w1.sum()

print(w1.eval())
print(s.eval())
print(s2.eval())
