
import pydot as pd
import theano
import theano.tensor as T
import numpy as np
from theano import shared


theano.config.floatX = 'float32'


a = T.fscalar()
k = T.iscalar()

kf = T.fscalar()
kf = k

temp, _ = theano.scan(
    fn = lambda r, ka: r * ka,
    non_sequences = kf,
    n_steps = k,
    outputs_info = 1.0
)

result = temp

fak = theano.function( [a, k], result, allow_input_downcast = True)

print(fak(2.0, 5.0))


print("all ok")