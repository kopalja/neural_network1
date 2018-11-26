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
    layers = None,
    # layers = (
    #     ConvLayer(input_shape = (1, 28, 28), output_images = 20, kernel_size = 5, activation_fn = T.nnet.relu),
    #     PoolLayer(size = 2),
    #     ConvLayer(input_shape = (20, 12, 12), output_images = 40, kernel_size = 5, activation_fn = T.nnet.relu),
    #     ConvLayer(input_shape = (40, 8, 8), output_images = 80, kernel_size = 3, activation_fn = T.nnet.relu),
    #     ConvLayer(input_shape = (80, 6, 6), output_images = 100, kernel_size = 3, activation_fn = T.nnet.relu),
    #     ConvLayer(input_shape = (100, 4, 4), output_images = 120, kernel_size = 3, activation_fn = T.nnet.relu),
    #     FullyConectedLayer(in_size = 120 * 2 * 2, out_size = 100, activation_fn = T.nnet.relu),
    #     FullyConectedLayer(in_size = 100, out_size = 10, activation_fn = T.nnet.softmax)
    # ),
    training_data = training_data, 
    validation_data = validation_data, 
    cost_fn = Cf.probability,
    learning_rate = 0.03, 
    minibatch_size = 32, 
    dropout = 0.0,
    regulation_param = 0.1,
    update_type = Update.Vanilla,
    normalize_data = False,
    load_from_file = 'net.json',
    save_to_file = 'net.json'
)

end = time.time()
print("compilation time ", end - start)
print("==================")
#net.train(epoch = 10)
net.test()



