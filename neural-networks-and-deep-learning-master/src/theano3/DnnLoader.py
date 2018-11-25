import numpy as np
import theano
import theano.tensor as T
import json

from ConvLayer import ConvLayer
from PoolLayer import PoolLayer
from FullyConectedLayer import FullyConectedLayer


class DnnLoader(object):
    """2DConvolution, Pool and Full layers are supported"""

    @staticmethod
    def Save(filename, layers):
        if (filename == 'none'):
            return
        weights = [np.asarray(l.w.eval()) for l in layers]
        biases = [np.asarray(l.b.eval()) for l in layers]       
        layers_type = []
        ctor_params = []
        for layer in layers:
            ctor_params.append(layer.ctor_params)
            if type(layer) is ConvLayer:
                layers_type.append('Convolution')
            elif type(layer) is FullyConectedLayer:
                layers_type.append('Full')
            elif type(layer) is PoolLayer:
                layers_type.append('Pool')
        data = {
            "layers": layers_type,
            "inputparams": ctor_params,
            "weights": [w.tolist() for w in weights],
            "biases": [b.tolist() for b in biases]
        }
        f = open(filename, "w")
        json.dump(data, f)
        f.close()

    @staticmethod
    def Load(filename):
        f = open(filename, "r")
        data = json.load(f)
        f.close()
        weights = [np.array(w) for w in data["weights"]]
        biases = [np.array(b) for b in data["biases"]]
        layers = []
        for layer, params, w, b in zip(data["layers"], data["inputparams"], weights, biases):
            newLayer = 0
            if (params[len(params) - 1] == 'relu'):
                params[len(params) - 1] = T.nnet.relu
            elif (params[len(params) - 1] == 'sigmoid'):
                params[len(params) - 1] = T.nnet.sigmoid
            elif (params[len(params) - 1] == 'softmax'):
                params[len(params) - 1] = T.nnet.softmax

            if layer == "Convolution":
                newLayer = ConvLayer(params[0], params[1], params[2], params[3])
            elif layer == "Full":
                newLayer = FullyConectedLayer(params[0], params[1], params[2])
            elif layer == "Pool":
                newLayer = PoolLayer(params[0])
            newLayer.w = theano.shared(np.asarray(w, theano.config.floatX), borrow=True) 
            newLayer.b = theano.shared(np.asarray(b, theano.config.floatX), borrow=True)
            if (layer == "Convolution" or layer == "Full"):
                newLayer.params = [newLayer.w, newLayer.b]
            layers.append(newLayer)
        return layers