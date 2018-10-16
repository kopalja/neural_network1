
import theano
import theano.tensor as T


def Softmax(output_batch):
    sum_batch, _ = theano.scan(fn = lambda output_vector: T.sum(T.exp(output_vector)), sequences = output_batch)
    out, _ = theano.scan(fn = lambda output_vector, summ: T.exp(output_vector) / summ, sequences = [output_batch, sum_batch])
    return out

def ReLU(x): 
    return T.maximum(0.0, x)

def Sigmoid(x):
    return 1.0 / ( 1.0 + T.exp(-x) ) 