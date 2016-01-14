import json
import numpy as np
import cv2
import os
import time
import re
import csv


class EnvironmentClassifier(object):

	def __init__(self, ):
		pass


	def getWords(self, text_input):
		return re.compile('\w+').findall(text_input)

	def get_category_list(self, category_path):
		scene_category_list = []
		with open(category_path, 'rb') as f:
			reader = csv.reader(f)
			for row in reader:
				scene_category_list.append(self.getWords(row[0])[1])
		self.scene_category_list = scene_category_list

	def deploy_network(self, MODEL_FILE, PRETRAINED, MEAN_FILE):
		caffe_root = '../../caffe/'
		import sys
		sys.path.insert(0,caffe_root + 'python')

		import caffe

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
		self.net = net
		self.transformer = transformer

	def video_reader(self, VIDEOPATH):

		#read video anmd get video parameters
		cap = cv2.VideoCapture(VIDEOPATH)
		self.length = int(cap.get(cv2.cv.CV_CAP_PROP_FRAME_COUNT))
		self.width  = int(cap.get(cv2.cv.CV_CAP_PROP_FRAME_WIDTH))
		self.height = int(cap.get(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT))
		self.fps    = cap.get(cv2.cv.CV_CAP_PROP_FPS)
		self.cap = cap
		print "movie length is {}, and frame rate is {}".format(self.length, self.fps)

	def set_timestamp(self, start_time = 0, skipping_frame = 500):
		num_frame = self.data_array.shape[0]
		step = int(1./skipping_frame * 10**6)
		timestamp = [start_time + x *step for x in range(num_frame)]
		return timestamp

	def environment_classification(self, frame_index = 1, skipping_frame = 500):
		data = []
		while(self.cap.isOpened()):
			self.cap.set(cv2.cv.CV_CAP_PROP_POS_FRAMES, frame_index)
			ret, frame = self.cap.read()
			if frame == None or frame_index >= self.length:
				break
			open_cv_frame = np.asarray(frame).astype(np.float32)
			movie_frame = np.divide(open_cv_frame,255)

			self.net.blobs['data'].data[...] = self.transformer.preprocess('data', movie_frame)
			out = self.net.forward()
			prediction_array = np.asarray(out['prob'][0])
			prediction_index = out['prob'][0].argmax()
			data.append(prediction_array)
			print np.asarray(data)
			print "Predicted class is {} {}, with a likelyhood of {}.".format(out['prob'][0].argmax(), self.scene_category_list[prediction_index], out['prob'][0][prediction_index])
			frame_index += skipping_frame

		self.data_array = np.asarray(data)
		return self.data_array
