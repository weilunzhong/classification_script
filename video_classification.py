#-*- coding: UTF-8 -*-
# coding=gbk

import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import json
import csv
import time
start_time = time.time()

caffe_root = '../../caffe/'
import sys
sys.path.insert(0,caffe_root + 'python')


import caffe

def getIdAndPath (id_vector, path_vector):
  file_id = id_vector.pop()
  source_path = path_vector.pop()
  return file_id, source_path

movie_counter = 0

log_path = '/home/vionlabs/Documents/scene_classification/scene_log/'


# MODEL_FILE = '../../googlenet_places205/deploy_places205.prototxt'
# PRETRAINED = '../../googlenet_places205/googlelet_places205_train_iter_2400000.caffemodel'
MODEL_FILE = '/home/vionlabs/Documents/scene_classification/Places_CNDS_model/deploy.prototxt'
PRETRAINED = '/home/vionlabs/Documents/scene_classification/Places_CNDS_model/8conv3fc_DSN.caffemodel'
# MODEL_FILE = '../../placesCNN_upgraded/places205CNN_deploy_upgraded.prototxt'
# PRETRAINED = '../../placesCNN_upgraded/places205CNN_iter_300000_upgraded.caffemodel'

#VIDEOPATH = '/home/vionlabs/Documents/weilun_thesis/caffe/examples/test_data/Life_of_Pi.avi'
#VIDEOPATH = '/home/vionlabs/Documents/scene_classification/video_file/welcome_to_sweden.mp4'

list_of_movies = open('video_files_for_computing.txt', 'r')
movies = json.load(list_of_movies)

for movie_id in movies:
  file_id = movie_id
  VIDEOPATH = movies[file_id]
  print file_id, VIDEOPATH
  if os.path.isfile(log_path + movie_id +'.txt'):
    movie_counter += 1
    continue
  scenelog = open("../scene_log/" + file_id + ".txt", "w")


  #caffe.set_phase_test()
  caffe.set_mode_gpu()
  net = caffe.Classifier(MODEL_FILE, PRETRAINED,
                         mean=np.load('../places_models/places205_mean.').mean(1).mean(1),
                         channel_swap=(0,1,2),
                         raw_scale=255,
                         image_dims=(256, 256))

  #cap = cv2.VideoCapture('welcome_to_sweden.mp4')
  #cap = cv2.VideoCapture('homeland.mkv')
  print file_id, VIDEOPATH.encode('utf8')
  cap = cv2.VideoCapture(VIDEOPATH.encode('utf8'))

  print movie_counter, movie_id, 'finished'
  frame_index = 1
  frame_limit = 500
  while frame_index < frame_limit:
      frame_index += 1
      ret, frame = cap.read()
      if frame == None:
        print '!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!'
        break
      if frame_index == 160:
        # if frame == None:
        #   print 'got it'
        # print frame.shape
  #      cv2.imwrite('welcome_to_sweden.jpg', frame)
  #      cv2.imshow('frame',frame)
        #print '--------------------------------'
        open_cv_frame = np.asarray(frame).astype(np.float32)
        movie_frame = np.divide(open_cv_frame,255)
        #input_image = caffe.io.load_image(movie_frame)
        #cv2.imshow('frame', movie_frame)
        #print movie_frame.shape, 'shape of frame'
        prediction = net.predict([movie_frame])
        print max(prediction[0])
        result = prediction[0]
        m = max(result)
        index = result.argmax(axis=0)
        #print 'Category number:', index
        #with open('../../placesCNN/category_filter.csv', 'rb') as f:
        with open('categoryIndex_places205.csv', 'rb') as f:
          reader = csv.reader(f)
          for i, row in enumerate(reader):
            if i == index:
              #print row[0]
              scenelog.write(str(index) + os.linesep)
              scenelog.write(str(row[0]) + os.linesep)
              scenelog.write(str(m) + os.linesep)
            i += 1
        #print time.clock() - start_time, "seconds"
      if frame_index == frame_limit-1:
        frame_index = 1
        continue
      #if cv2.waitKey(1) & 0xFF == ord('q'):
      #    break

  print time.time() - start_time, "seconds"
  scenelog.close()
  movie_counter += 1
  print movie_counter, movie_id, 'finished'
  print '-----------------------'

  #  cap.release()
  #  cv2.destroyAllWindows()
