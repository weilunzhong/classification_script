import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import csv
import time
import re

caffe_root = '../../caffe/'
import sys
sys.path.insert(0,caffe_root + 'python')

import caffe

def getWords(text_input):
  return re.compile('\w+').findall(text_input)

#scenelog = open("casino_royale.txt", "w")
MODEL_FILE = '/home/vionlabs/Documents/scene_classification/places_models/vgg16_places2/deploy.prototxt'
PRETRAINED = '/home/vionlabs/Documents/scene_classification/places_models/vgg16_places2/vgg16_places2.caffemodel'
#IMAGE_PATH = '../examples/test_data/test_image'
#IMAGE_PATH = '../../../indoor_image/dining_room'
VIDEOPATH = '../video_file/welcome_to_sweden.mp4'


# caffe.set_phase_test()
caffe.set_mode_gpu()
net = caffe.Classifier(MODEL_FILE, PRETRAINED,
                       mean=np.load( '../places_models/Places_CNDS_model/places_mean.npy').mean(1).mean(1),
                       channel_swap=(0,1,2),
                       raw_scale=256,
                       image_dims=(224, 224))

#cap = cv2.VideoCapture('welcome_to_sweden.mp4')
#cap = cv2.VideoCapture('homeland.mkv')
cap = cv2.VideoCapture(VIDEOPATH)

scene_category_list = []
with open('/home/vionlabs/Documents/scene_classification/places_models/vgg16_places2/categories.txt', 'rb') as f:
  reader = csv.reader(f)
  for row in reader:
    scene_category_list.append(getWords(row[0])[1])

print scene_category_list

frame_index = 1
frame_limit = 500
while frame_index < frame_limit:
    frame_index += 1
    ret, frame = cap.read()
    if frame_index == 60:
#      cv2.imwrite('welcome_to_sweden.jpg', frame)
      cv2.imshow('frame',frame)
      print '--------------------------------'
      open_cv_frame = np.asarray(frame).astype(np.float32)
      movie_frame = np.divide(open_cv_frame,255)
      #input_image = caffe.io.load_image(movie_frame)
      prediction = net.predict([movie_frame])
      print max(prediction[0])
      result = prediction[0]
      m = max(result)
      index = result.argmax(axis=0)
      cv2.putText(frame, scene_category_list[index], (15,50), cv2.FONT_HERSHEY_SIMPLEX, 1, cv2.cv.CV_RGB(0,255,0))
      #cv2.imshow('frame', frame)
      print 'Category number:', index
      # print row[0]
      # scenelog.write(str(index) + os.linesep)
      # scenelog.write(str(row[0]) + os.linesep)
      # scenelog.write(str(m) + os.linesep)
      frame_index += 1
    if frame_index == frame_limit-1:
      frame_index = 1
      continue
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
