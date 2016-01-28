from vgg16_places2 import VggPlaces2
import tensorflow as tf
import cv2
import numpy as np


img = cv2.imread('/home/vionlabs/Downloads/test.jpg')
img = cv2.resize(img, PlacesNet.scale_size)
img = img.astype(np.float32)
print img.shape

def image_reader(path):
    img = cv2.imread(path)
    img = cv2.resize(img, PlacesNet.scale_size)
    img = img.astype(np.float32)
    return img

def batches(batch_size, path_list):
    images = np.concatenate(map(image_reader, path_list))
    yield images


#test_data = tf.placeholder(tf.float32, shape(batch_size, 224,224,3) 
#tf.reshape(tf.identity(img), [1]+list(img.shape)) 
#test_data = tf.placeholder(tf.float32, shape=(1,224,224,3))
img_path = ["~/Downloads/test.jpg",  "~/Downloads/test_1.jpg"]

    


#net = PlacesNet({'data':test_data})

with tf.Session() as sesh:
    # Load the data
    net.load('places205_caffemodel.npy', sesh)
    for images in enumerate(batches(2, img_path)):
        output = sesh.run(net.get_output())
        print output.shape
    # Forward pass
    #output = sesh.run(net.get_output())
    #print output.shape
