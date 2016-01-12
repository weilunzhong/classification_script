import numpy as np
import cv2
import matplotlib.pyplot as plt
import os
import time
import re
import csv

caffe_root = '../../caffe/'
import sys
sys.path.insert(0,caffe_root + 'python')

import caffe
def getWords(text_input):
	return re.compile('\w+').findall(text_input)

#path for all the files needed
MODEL_FILE = '/home/vionlabs/Documents/scene_classification/places_models/vgg16_places2/deploy.prototxt'
PRETRAINED = '/home/vionlabs/Documents/scene_classification/places_models/vgg16_places2/vgg16_places2.caffemodel'
VIDEOPATH = '../video_file/welcome_to_sweden.mp4'
MEAN_FILE = '../places_models/Places_CNDS_model/places_mean.npy'

if not os.path.isfile(MODEL_FILE):
	print "pretrained model not in place"

caffe.set_mode_gpu()
net = caffe.Classifier(MODEL_FILE, PRETRAINED, caffe.TEST)

transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
transformer.set_transpose('data', (2,0,1))
transformer.set_mean('data', np.load(MEAN_FILE).mean(1).mean(1))
transformer.set_raw_scale('data', 255)
transformer.set_channel_swap('data',(0,1,2))

#with a batch size of 50
net.blobs['data'].reshape(50, 3, 224, 224)



#read video anmd get video parameters
cap = cv2.VideoCapture(VIDEOPATH)
length = int(cap.get(cv2.cv.CV_CAP_PROP_FRAME_COUNT))
width  = int(cap.get(cv2.cv.CV_CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT))
fps    = cap.get(cv2.cv.CV_CAP_PROP_FPS)
print "movie length is {}, and frame rate is {}".format(length, fps)


frame_index = 6000

scene_category_list = []
with open('/home/vionlabs/Documents/scene_classification/places_models/vgg16_places2/categories.txt', 'rb') as f:
  reader = csv.reader(f)
  for row in reader:
    scene_category_list.append(getWords(row[0])[1])

print scene_category_list


while(cap.isOpened()):
	cap.set(cv2.cv.CV_CAP_PROP_POS_FRAMES, frame_index)
	ret, frame = cap.read()
	if frame ==None:
		break
	open_cv_frame = np.asarray(frame).astype(np.float32)
	movie_frame = np.divide(open_cv_frame,255)

	net.blobs['data'].data[...] = transformer.preprocess('data', movie_frame)
	out = net.forward()
	prediction_index = out['prob'][0].argmax()

	print "Predicted class is {} {}, with a likelyhood of {}.".format(out['prob'][0].argmax(), scene_category_list[prediction_index], out['prob'][0][prediction_index])
	frame_index += 500
	
	cv2.putText(frame, scene_category_list[prediction_index], (15,50), cv2.FONT_HERSHEY_SIMPLEX, 1, cv2.cv.CV_RGB(0,255,0))
	cv2.imshow('frame', frame)


	if cv2.waitKey(1) & 0xFF == ord('q'):
		break

cap.release()
cv2.destroyAllWindows()





