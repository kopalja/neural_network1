import imageio
import numpy as np
from Network3 import *

def open_image(img_path: str, as_gray = True):
    img = imageio.imread(img_path, as_gray = as_gray)
    return img.astype('bool')  if as_gray else img

def reconstruct(tiles: np.ndarray):
    width, height = tiles.shape[0] * tiles.shape[2], tiles.shape[1] * tiles.shape[3]
    return tiles.swapaxes(1,2).reshape(width, height)

def get_tiles(img: np.ndarray, tile_size = 10):
    print(img.shape)
    _nrows, _ncols = img.shape
    _size = img.size
    _strides = img.strides

    nrows, _m = divmod(_nrows, tile_size)
    ncols, _n = divmod(_ncols, tile_size)
    if _m != 0 or _n != 0:
        return None


    return np.lib.stride_tricks.as_strided(
        np.ravel(img),
        shape=(nrows, ncols, tile_size, tile_size),
        strides=(tile_size * _strides[0], tile_size * _strides[1], *_strides),
        writeable=True
    )

def get_translations(tile: np.ndarray, max_translation):
    []

def translate(arr: np.ndarray, x, y):
    result = np.empty_like(arr)
    if y > 0:
        result[:y] = 0
        result[y:] = arr[:-y]
    elif y < 0:
        result[y:] = 0
        result[:y] = arr[-y:]
    else:
        result = arr

    res = np.empty_like(result)

    if x > 0:
        res[:,:x] = 0
        res[:,x:] = result[:,:-x]
    elif x < 0:
        res[:,x:] = 0
        res[:,:x] = result[:,-x:]
    else:
        res = result
    return res

def save(bits, path):
    imageio.imsave(path, bits)


def are_tiles_same(t1, t2):
    for i in range(t1.size):
        if (t1[i] != t2[i]):
            return False
    return True







def shift(tile, x_axis, n):
    new_tile = np.copy(tile)
    if x_axis:
        for i in range(tile.shape[0]):
            for j in range(tile.shape[1]):
                if i + n >= 0 and j + n < tile.shape[1]:
                    new_tile[i][j + n] = tile[i][j]
    else:
          for i in range(tile.shape[0]):
                for j in range(tile.shape[1]):
                if i + n >= 0 and j + n < tile.shape[1]:
                    new_tile[i][j + n] = tile[i][j]      
                 
    return new_tile






img = open_image('house2.bmp', True)
tiles = get_tiles(img)


######1
shifted_tiles = np.zeros(shape = (tiles.shape[0], tiles.shape[1], 10, tiles.shape[2], tiles.shape[3]))
# create shifts tiles
i = 0
j = 0
for row in tiles:
    for col in row:
        shifts = []
        for k in range(-2, 3):
            shifts.append(shift(col, True, k))
            shifts.append(shift(col, False, k))
        print(shifted_tiles.shape)
        shifted_tiles[i][j] = np.array(shifts)
        print('new')
        print(shifted_tiles[i][j])
        j += 1
    j = 0
    i += 1

inpt = np.zeros(shape = (tiles.shape[0] * tiles.shape[1], 10, tiles.shape[2] * tiles.shape[3]))
print(inpt.shape)
i = 0
for row in shifted_tiles:
    for col in row:
        shifted = []
        for k in col:
            sample = k.ravel()
            shifted.append(sample)
        inpt[i] = np.array(shifted)
        i += 1
data = (inpt, inpt)

print('a')
print(inpt[0])
######1

######2
# inpt = np.zeros(shape = (tiles.shape[0] * tiles.shape[1], tiles.shape[2] * tiles.shape[3]))
# output = np.zeros(shape = (tiles.shape[0] * tiles.shape[1], tiles.shape[2] * tiles.shape[3]))
# print(inpt.shape)
# i = 0
# for row in tiles:
#     for col in row:
#         sample = col.ravel()
#         inpt[i] = sample
#         output[i] = sample
#         i += 1
# data = (inpt, output)
######2

net = Batch_Network(
    layers = (
        FullyConectedLayer(in_size = 100, out_size = 8, activation_fn = T.nnet.sigmoid),
        FullyConectedLayer(in_size = 8, out_size = 100, activation_fn = T.nnet.sigmoid)
    ),
    training_data = data,
    validation_data = data, 
    cost_fn = Cf.quadratic,
    learning_rate = 0.05, 
    minibatch_size = 1, 
    dropout = 0.0,
    l2_regulation = 0.1,
    update_type = Update.Sgd
)

# set network outputs to 0 or 1
network_outputs = net.train(epoch = 200)
for network_output in network_outputs:
    for i in range(network_output.size):
        network_output[i] = 0 if network_output[i] < 0.5 else 1



number_of_tiles = 0
for i in range(len(network_outputs)):
    for j in range(i + 1, len(network_outputs)):
        if are_tiles_same(network_outputs[i], network_outputs[j]) == False:
                number_of_tiles += 1
print("number of tiles : ", number_of_tiles / 2)


# copy network outputs into tails
tiles = np.zeros(tiles.shape)
index = 0
for i in range(tiles.shape[0]):
    for j in range(tiles.shape[1]):
        tiles[i][j] = network_outputs[index].reshape((10, 10))
        index += 1


# tiles array into image
bits = reconstruct(tiles)
save(bits, "out.bmp")

        


