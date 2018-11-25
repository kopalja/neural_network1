import time
start = time.time()

from Network3 import *
from DataLoader import *
import numpy as np
#### Virat
### Visor
### http://clickdamage.com/sourcecode/cv_datasets.php



training_data, validation_data, test_data = load_data_shared()
expanded_data, _, _ = load_data_shared("../data/mnist_expanded.pkl.gz")



net = Batch_Network(
    layers = (
        ConvLayer(input_shape = (1, 28, 28), output_images = 20, kernel_size = 5, activation_fn = T.nnet.relu),
        PoolLayer(size = 2),
        ConvLayer(input_shape = (20, 12, 12), output_images = 40, kernel_size = 5, activation_fn = T.nnet.relu),
        PoolLayer(size = 2),
        FullyConectedLayer(in_size = 40 * 4 * 4, out_size = 1000, activation_fn = T.nnet.relu),
        FullyConectedLayer(in_size = 1000, out_size = 1000, activation_fn = T.nnet.relu),
        FullyConectedLayer(in_size = 1000, out_size = 10, activation_fn = T.nnet.softmax)
    ),
    training_data = training_data, 
    validation_data = validation_data, 
    cost_fn = Cf.probability,
    learning_rate = 0.05, 
    minibatch_size = 64, 
    dropout = 0.5,
    regulation_param = 0.1,
    update_type = Update.Vanilla,
    normalize_data = True,
    load_from_file = 'none',
    save_to_file = 'net.json'
)

end = time.time()
print("compilation time ", end - start)
print("==================")
net.train(epoch = 2)