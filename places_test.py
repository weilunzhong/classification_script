from places_tensor import PlacesNet
import tensorflow as tf
import cv2
import numpy as np


img = cv2.imread('/home/vionlabs/Downloads/test.jpg')
img = cv2.resize(img, PlacesNet.scale_size)
img = img.astype(np.float32)
print img.shape


test_data = tf.reshape(tf.identity(img), [1]+list(img.shape))

#test_data = tf.placeholder(tf.float32, shape=(1,224,224,3))

net = PlacesNet({'data':test_data})

with tf.Session() as sesh:
    # Load the data
    net.load('places205_caffemodel.npy', sesh)
    # Forward pass
    output = sesh.run(net.get_output())
    print output