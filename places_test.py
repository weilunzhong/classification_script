from vgg16_places2 import VggPlaces2
import tensorflow as tf
import cv2
import numpy as np
from vionaux.rnd import vidioids

#img = cv2.imread('/home/vionlabs/Downloads/test.jpg')
#img = cv2.resize(img, VggPlaces2.scale_size)
#img = img.astype(np.float32)
#print img.shape

class EnvironmentClassifier(object):
        
    def batches(self):
        images = map(image_reader, path_list)
        for image in images :
            yield image


    def net_deployment(self):
        test_data = tf.placeholder(tf.float32, shape=(2,224,224,3))
        net = VggPlaces2({'data':test_data})

        with tf.Session() as sesh:
            probs = net.get_output()
            test_labels = tf.placeholder(tf.int32, shape=(2))
            top_k_op = tf.nn.in_top_k(probs, test_labels, 1)
            # load the data
            net.load('vgg16_places2_caffemodel.npy', sesh)
            #add the 4th dimension to make tensor work
            images_list = [np.expand_dims(i,axis=0) for i in batches(img_path)]
            images = np.concatenate(images_list, axis=0)
            #iamges = tf.reshape(tf.identity(images), [1]+list(images.shape)) 
            output = sesh.run(top_k_op,feed_dict={test_data: images, test_labels: np.empty(2)})
            print output
            


if "__name__" == "__main__":
    video_path = "/mnt/movies03/boxer_movies/tt3247714/Survivor (2015)/Survivor.2015.720p.BluRay.x264.YIFY.mp4"
    VHH = VionVideoHandler()
    g = VHH.get_frame(video_path, 0.01, 4000, 6000)
    for frame in g:
        print x[0].shape, x[1]
