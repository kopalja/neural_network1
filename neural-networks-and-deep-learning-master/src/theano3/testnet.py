
from Network3 import *
from DataLoader import *
import numpy as np
#### Virat
### Visor
### http://clickdamage.com/sourcecode/cv_datasets.php
training_data, validation_data, test_data = load_data()
expanded_data, _, _ = load_data("../data/mnist_expanded.pkl.gz")







net = Batch_Network(
    layers = LayersWrapper(
        input_shape = (1, 28, 28),
        layers_description = (
            ConvLayer_w(output_images = 20, kernel_size = 5, activation_fn = T.nnet.relu),
            PoolLayer_w(shape = (2, 2)),
            ConvLayer_w(output_images = 40, kernel_size = 5, activation_fn = T.nnet.relu),
            PoolLayer_w(shape = (2, 2)),
            FullyConectedLayer_w(size = 1000, activation_fn = T.nnet.relu),    
            FullyConectedLayer_w(size = 1000, activation_fn = T.nnet.relu),
            FullyConectedLayer_w(size = 10, activation_fn = T.nnet.softmax),      

        )
    ),
    # layers = (
    #     ConvLayer(input_shape = (1, 28, 28), output_images = 20, kernel_size = 5, activation_fn = T.nnet.relu),
    #     PoolLayer(shape = (2, 2)),
    #     ConvLayer(input_shape = (20, 12, 12), output_images = 40, kernel_size = 5, activation_fn = T.nnet.relu),
    #     PoolLayer(shape = (2, 2)),
    #     FullyConectedLayer(in_size = 40 * 4 * 4, out_size = 100, activation_fn = T.nnet.relu),
    #     FullyConectedLayer(in_size = 100, out_size = 10, activation_fn = T.nnet.softmax),
    # ),
    
    training_data = training_data,
    validation_data = validation_data, 
    cost_fn = Cf.probability,
    learning_rate = 0.003, 
    minibatch_size = 100, 
    dropout = 0.5,
    l2_regulation = 0.1,
    update_type = Update.Adam,
    normalize_data = False,
    load_from_file = None,
    save_to_file = 'net.json'
)

    
# end = time.time()
# print("compilation time ", end - start)
# print('==================================')
net.train(epoch = 15)
#net.test()