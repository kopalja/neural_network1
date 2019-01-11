
import numpy as np
import urllib.request
import string
import os
from shutil import copyfile
import cv2
from os import listdir
from os.path import isfile, join
import random
import gzip, pickle
from PIL import Image
import matplotlib.pyplot as plt

def dowload_images(path):
    text = open(path, 'r')
    counter = 5415
    for line in text:
        source = line
        destination = "C:\\Users\\kopi\\Desktop\\snakes_urls\\cats\\cat", str(counter), ".jpg"
        destination = ''.join(destination)
        try:
            urllib.request.urlretrieve(source, destination)
            size = os.path.getsize(destination)
            if size < 10000:
                os.remove(destination)
            else:
                counter += 1
        except Exception:
            print("problem", counter)

def delete_empty_images():
    path = str("C:\\Users\kopi\\Desktop\\snakes_urls\\")
    from os import listdir
    from os.path import isfile, join
    onlyfiles = [f for f in listdir(path) if isfile(join(path, f))]
    for name in onlyfiles:
        img_path = path + name
        size = os.path.getsize(img_path)
        if size == 2051:
            os.remove(img_path)
        print(size)


def select_and_sort(path):
    snakes_path = path + "snakes\\"
    onlyfiles = [f for f in listdir(path) if isfile(join(path, f))]
    counter = 4703
    for img_name in onlyfiles:
        img_path = path + img_name
        destination_path = snakes_path + "snake" + str(counter) + ".jpg"
        size = os.path.getsize(img_path)
        if size > 10000:
            copyfile(img_path, destination_path)
            counter += 1
        os.remove(img_path)

############################################################################################################################
def crop_from_corner(corner, img, width, height):
    img_height, img_width, channels = img.shape
    if corner == 0:
        return img[0:height, 0:width]
    elif corner == 1:
        return img[0:height, img_width - width:img_width]
    elif corner == 2:
        return img[img_height - height:img_height, 0:width]
    else:
        return img[img_height - height:img_height, img_width - width:img_width]


# Create images with same sizes from origin (downloaded) images.
def make_same_size(width, height, source_dir_path, destination_path, categori):
    onlyfiles = [f for f in listdir(source_dir_path) if isfile(join(source_dir_path, f))]
    counter = 0
    for img_name in onlyfiles:
        img_path = source_dir_path + img_name
        img = cv2.imread(img_path)
        if type(img) == type(None):
            continue
        img_height, img_width, _ = img.shape
        if img_height > height and img_width > width and img_height < 1.5 * height and img_width < 1.5 * width:
            for  i in range(4):
                croppped_img = crop_from_corner(i, img, width, height)
                write_path = destination_path + categori + str(counter) + '.jpg'
                cv2.imwrite(write_path, croppped_img)
                counter += 1
############################################################################################################################

def save_data_pictures(path, cats_and_dogs):
    category1_folder_path = path + 'dogs\\'
    if cats_and_dogs:
        category2_folder_path = path + 'cats\\' 
    else:
        category2_folder_path = path + 'snakes\\'

    category1 = _load_category(category1_folder_path)
    category2 = _load_category(category2_folder_path)
    output1 = [np.array([1.0, 0.0]) for i in range(len(category1))]
    output2 = [np.array([0.0, 1.0]) for i in range(len(category2))]
    inputs = category1 + category2
    outputs = output1 + output2
    data = list(zip(inputs, outputs))
    random.shuffle(data)
    inputs = [d[0] for d in data]
    outputs = [d[1] for d in data]
    training_data = [inputs[0:len(data) - 2000], outputs[0:len(data) - 2000]]
    validation_data = [inputs[len(data) - 2000:len(data)], outputs[len(data) - 2000:len(data)]]

    result = [training_data, validation_data]
    file_name = path + 'dogs_and_snakes.pkl'
    output_file = open(file_name, 'wb')
    pickle.dump(result, output_file)





def _load_category(folder_path):
    images = [f for f in listdir(folder_path) if isfile(join(folder_path, f))]
    data = []
    for image in images:
        image_path = folder_path + image
        img = cv2.imread(image_path)
        r = img[..., 0].ravel()
        g = img[..., 1].ravel()
        b = img[..., 2].ravel()
        data.append(np.concatenate((r, g, b)))
    return data
############################################################################################################################


save_data_pictures("C:\\Users\\kopi\\Desktop\\snakes_urls\\", False)



#dowload_images("C:\\Users\\kopi\\Desktop\\snakes_urls\\url8.txt")
#delete_empty_images()

#select_and_sort("C:\\Users\\kopi\\Desktop\\snakes_urls\\")

#make_same_size(400, 300, "C:\\Users\\kopi\\Desktop\\snakes_urls\\snakes_origin\\", "C:\\Users\\kopi\\Desktop\\snakes_urls\\snakes\\", 'snake')

# f = open("C:\\Users\\kopi\\Desktop\\snakes_urls\\dogs_and_cats.pkl", 'rb')
# training, validation = pickle.load(f, encoding='latin1')



# inputs = training[0]
# outputs = training[1]

# print(len(inputs))

# def test():
#     l1 = [1, 2, 3]
#     l2 = [5, 5, 5]
#     z = list(zip(l1, l2))
#     return z



# ar = [[255, 255, 255, 255, 255, 255, 255, 255], [0, 0, 0, 0, 0, 0, 0, 0],
#         [255, 255, 255, 255, 255, 255, 255, 255], [0, 0, 0, 0, 0, 0, 0, 0],
#         [255, 255, 255, 255, 255, 255, 255, 255], [0, 0, 0, 0, 0, 0, 0, 0],
#         [255, 255, 255, 255, 255, 255, 255, 255], [0, 0, 0, 0, 0, 0, 0, 0],
#         [255, 255, 255, 255, 255, 255, 255, 255], [0, 0, 0, 0, 0, 0, 0, 0],
#         [255, 255, 255, 255, 255, 255, 255, 255], [0, 0, 0, 0, 0, 0, 0, 0],]
# ar = np.asarray(ar)


# plt.imshow(ar, cmap="gray")
# plt.show()