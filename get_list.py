#-*- coding: UTF-8 -*-
# coding=gbk


import json

# list_of_movies = open('video_files_for_computing.txt', 'r')
# movies = json.load(list_of_movies)
# index = 0
# for element in movies:
# 	index += 1
# 	print movies[element]

# print index

# print movies['tt0484562']

# string = open('string.txt', 'r')
# data = json.load(string)
# print 'fds'

# for element in data:
# 	print element, data[element]


import os
path = '/home/vionlabs/Documents/scene_classification/scene_log/'
if os.path.isfile(path + '123456.txt'):
	print 'haha'