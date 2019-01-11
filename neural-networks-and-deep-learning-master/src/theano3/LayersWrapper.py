import numpy as np

from ConvLayer import ConvLayer
from PoolLayer import PoolLayer
from FullyConectedLayer import FullyConectedLayer

class LayersWrapper:
    def __init__(self, input_shape, layers_description):
        self.input_shape = input_shape
        self.layers_description = layers_description
    
    def create_layers(self):
        next_input_shape = self.input_shape
        layers = []
        for layer in self.layers_description:
            if type(layer) == ConvLayer_w:
                layers.append(ConvLayer(
                    input_shape = next_input_shape,
                    output_images = layer.output_images,
                    kernel_size = layer.kernel_size,
                    activation_fn = layer.activation_fn
                ))
                next_input_shape = (layer.output_images, next_input_shape[1] - layer.kernel_size + 1, next_input_shape[2] - layer.kernel_size + 1)
            elif type(layer) == PoolLayer_w:
                layers.append(PoolLayer(
                    shape = layer.shape
                ))
                next_input_shape = (next_input_shape[0], int(next_input_shape[1] // layer.shape[0]), int(next_input_shape[2] // layer.shape[1]))
            elif type(layer) == FullyConectedLayer_w:
                layers.append(FullyConectedLayer(
                    in_size = int(np.prod(next_input_shape)),
                    out_size = layer.size,
                    activation_fn = layer.activation_fn
                )) 
                next_input_shape = (layer.size,)
        return layers
        


class ConvLayer_w:
    def __init__(self, output_images, kernel_size, activation_fn):
        self.output_images = output_images
        self.kernel_size = kernel_size
        self.activation_fn = activation_fn

class PoolLayer_w:
    def __init__(self, shape):
        self.shape = shape


class FullyConectedLayer_w:
    def __init__(self, size, activation_fn):
        self.size = size
        self.activation_fn = activation_fn