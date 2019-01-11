
from Network3 import *
from DataLoader import load_data
from LayersWrapper import *
import numpy as np
import pickle
import matplotlib.pyplot as plt
#### Virat
### Visor
### http://clickdamage.com/sourcecode/cv_datasets.php



training_data, mnist_validation_data, test_data = load_data()
#expanded_data, _, _ = load_data("data/mnist_expanded.pkl.gz")
t11 = training_data[0]
t22 = training_data[1]
training_data = [t11[:500], t22[:500]]
# mnist = [t11[:200], t22[:200]]




f = open("C:\\Users\\kopi\\Desktop\\snakes_urls\\dogs_and_snakes.pkl", 'rb')
training_data, validation_data = pickle.load(f, encoding='latin1')
f.close()

t1 = training_data[0]
t2 = training_data[1]
t1 = [i[0:150 * 200] for i in t1]

for i in range(len(t1)):
    t1[i] = t1[i] / 255.0

print(len(t1))
training_data = [t1[:500], t2[:500]]

# t1 = validation_data[0]
# t2 = validation_data[1]
# t1 = [i[0:150*200] for i in t1]
# validation_data = [t1[:300], t2[:300]]


# ar = np.reshape(ar, newshape=(150, 200))
# plt.imshow(ar, cmap="gray")
# plt.show()



net = Batch_Network(
    layers = LayersWrapper(
        input_shape = (1, 150, 200),
        layers_description = (
            ConvLayer_w(output_images = 20, kernel_size = 5, activation_fn = T.nnet.sigmoid),
            PoolLayer_w(shape = (2, 2)),
            ConvLayer_w(output_images = 20, kernel_size = 5, activation_fn = T.nnet.sigmoid),
            PoolLayer_w(shape = (2, 2)),   
            #FullyConectedLayer_w(size = 10, activation_fn = T.nnet.sigmoid),   
            FullyConectedLayer_w(size = 100, activation_fn = T.nnet.sigmoid), 
            FullyConectedLayer_w(size = 2, activation_fn = T.nnet.sigmoid),      
        )
    ),
    training_data = training_data,
    validation_data = training_data, 
    cost_fn = Cf.crossEntropy,
    learning_rate = 0.3, 
    minibatch_size = 10, 
    dropout = 0.5,
    l2_regulation = 0.1,
    update_type = Update.Sgd,
    normalize_data = False,
    load_from_file = None,
    save_to_file = None
)
net.train(epoch = 1500)