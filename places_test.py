from vgg16_places2 import VggPlaces2
import tensorflow as tf
import cv2
import numpy as np


#img = cv2.imread('/home/vionlabs/Downloads/test.jpg')
#img = cv2.resize(img, VggPlaces2.scale_size)
#img = img.astype(np.float32)
#print img.shape

def image_reader(path):
    img = cv2.imread(path)
    print path
    img = cv2.resize(img, VggPlaces2.scale_size)
    img = img.astype(np.float32)
    return img



def batches(path_list):
    images = map(image_reader, path_list)
    for image in images :
        yield image



#images = batches(img_path)
#print type(images)
#for image in images:
#    print image.shape
#
#for idx, image in enumerate(batches(img_path)):
#    print image.shape
#



test_data = tf.placeholder(tf.float32, shape=(2,224,224,3))
img_path = ["/home/vionlabs/Downloads/test1.jpg",  "/home/vionlabs/Downloads/test.jpg"]


net = VggPlaces2({'data':test_data})

with tf.Session() as sesh:
    probs = net.get_output()
    test_labels = tf.placeholder(tf.int32, shape=(2))
    top_k_op = tf.nn.in_top_k(probs, test_labels, 1)
    # load the data
    net.load('vgg16_places2_caffemodel.npy', sesh)
    images_list = [np.expand_dims(i,axis=0) for i in batches(img_path)]
    images = np.concatenate(images_list, axis=0)
    #iamges = tf.reshape(tf.identity(images), [1]+list(images.shape)) 
    output = sesh.run(top_k_op,feed_dict={test_data: images, test_labels: np.empty(2)})
    print output
    # Forward pass
    #output = sesh.run(net.get_output())
    #print output.shape
