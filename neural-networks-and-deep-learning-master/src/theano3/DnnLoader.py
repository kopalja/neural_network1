import numpy as np
import theano
import theano.tensor as T
import json

from ConvLayer import ConvLayer
from PoolLayer import PoolLayer
from FullyConectedLayer import FullyConectedLayer


class DnnLoader(object):
    """ Layers : 2DConvolution, Pool and Full are supported \n
        Functions : relu, sigmoid, softmax are supported """

    @staticmethod
    def Save(filename, layers):
        if (filename == 'none'):
            return     
        layers_type, ctor_params, learned_parameters = [], [], []
        for layer in layers:
            if type(layer) is ConvLayer:
                layers_type.append('Convolution')
            elif type(layer) is FullyConectedLayer:
                layers_type.append('Full')
            elif type(layer) is PoolLayer:
                layers_type.append('Pool')
            ctor_params.append(layer.ctor_params)
            learned_parameters.append([np.array(p.eval()).tolist() for p in layer.params])
        data = {
            'layers': layers_type,
            'ctor_parameters': ctor_params,
            'learned_parameters': learned_parameters
        }
        f = open(filename, "w")
        json.dump(data, f)
        f.close()

    @staticmethod
    def Load(filename):
        f = open(filename, "r")
        data = json.load(f)
        f.close()
        DnnLoader.print_description(data['layers'], data['ctor_parameters'])
        loaded_layers = []
        for layer, ctor_parameters, learned_parameters in zip(data['layers'], data['ctor_parameters'], data['learned_parameters']):
            last = len(ctor_parameters) - 1
            if (ctor_parameters[last] == 'relu'):
                ctor_parameters[last] = T.nnet.relu
            elif (ctor_parameters[last] == 'sigmoid'):
                ctor_parameters[last] = T.nnet.sigmoid
            elif (ctor_parameters[last] == 'softmax'):
                ctor_parameters[last] = T.nnet.softmax

            if layer == 'Convolution':
                newLayer = ConvLayer(ctor_parameters[0], ctor_parameters[1], ctor_parameters[2], ctor_parameters[3], learned_parameters)
            elif layer == 'Full':
                newLayer = FullyConectedLayer(ctor_parameters[0], ctor_parameters[1], ctor_parameters[2], learned_parameters)
            elif layer == 'Pool':
                newLayer = PoolLayer(ctor_parameters[0])
            loaded_layers.append(newLayer)
        return loaded_layers

    
    @staticmethod
    def print_description(layers, params):
        for layer, params in zip(layers, params):
            print(layer, ": ")
            print("     input parameters", params)
