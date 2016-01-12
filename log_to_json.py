import json
import csv
import os
import math
import numpy as np
import re

def getWords(text):
	return re.compile('\w+').findall(text)


def getCategoryList(category_path):
	CATEGORY_NAMES = []
	with open(category_path, 'rb') as f:
		reader = csv.reader(f)
		for row in reader:
			full_name = getWords(row[0])
			#print full_name
			CATEGORY_NAMES.append(full_name[1])
		return CATEGORY_NAMES

def getSmallCategory(category_path):
	CATEGORY_NAMES = []
	with open(category_path, 'rb') as f:
		reader =csv.reader(f)
		for row in reader:
			#name = getWords(row)
			CATEGORY_NAMES.append(row[0])
		return CATEGORY_NAMES

def writeDataToClass(vec_of_scene_log, scene_name,scene_small, small_category):
	sequence = np.argsort(vec_of_scene_log)
	data_205 = []
	for element in sequence:
		data_205.append({scene_name[element]:vec_of_scene_log[element]})

	data_out_205 = {
	'movie id': movie_id,
	'frame number': sum(vec_of_scene_log),
	'scene frame number': sum(scene_small[1:28]),
	'indoor percentange': float(sum(scene_small[1:13]))/sum(scene_small[1:28]),
	'environment': data_205
	}
	
	sequence_small =np.argsort(scene_small)
	data_28 = []
	for element in sequence_small:
		data_28.append({small_category[element]: scene_small[element]})

	data_out_28 = {
	'movie id': movie_id,
	'frame number': sum(vec_of_scene_log),
	'scene frame number': sum(scene_small[1:28]),
	'indoor percentange': float(sum(scene_small[1:13]))/sum(scene_small[1:28]),
	'environment': data_28
	}
	return data_out_205, data_out_28


def writeToJson(data_205, data_28, movie_id):
	out_205_file = open('../Json/json_205/' + movie_id + '.json', 'w')
	json.dump(data_205, out_205_file)
	out_205_file.close()

	out_28_file = open('../Json/json_28/' + movie_id + '.json', 'w')
	json.dump(data_28, out_28_file)
	out_28_file.close()







category_path = '/home/vionlabs/Documents/scene_classification/category_list_flie/categoryIndex_places205.csv'
scene_small_path = '/home/vionlabs/Documents/scene_classification/category_list_flie/category_filter.csv'
small_vec_path = '/home/vionlabs/Documents/scene_classification/category_list_flie/list_of_catagory.csv'
json_folder_path = '/home/vionlabs/Documents/scene_classification/Json/'



list_of_movies = open('video_files_for_computing.txt', 'r')
movies = json.load(list_of_movies)
movie_counter = 0

for movie_id in movies:
	file_id = movie_id
	VIDEOPATH = movies[file_id]
	print movie_counter
	if os.path.isfile(json_folder_path + movie_id +'.json'):
		movie_counter += 1
		continue

	if os.path.isfile('/home/vionlabs/Documents/scene_classification/scene_log/' + file_id + ".txt"):
		source_path = '/home/vionlabs/Documents/scene_classification/scene_log/' + file_id + ".txt"
	else:
		continue



	#create json class for storing 
	data_205 =[]
	data_28 = []
	#print data

	#initialize the counter vector
	scene_vec = np.zeros(205,int)
	scene_small = np.zeros(28, int)

	#open source file
	scene_log = open(source_path, 'r')

	category = getCategoryList(category_path)
	small_category = getSmallCategory(small_vec_path)
	for i, line in enumerate(scene_log):
		if i%3 == 0:
			scene_vec[int(line)] += 1
			s = int(line)
			#print s
			with open(scene_small_path, 'rb') as file_of_small_category:
				reader = csv.reader(file_of_small_category)
				for index,row in enumerate(reader):
					if index == (2*s + 1):
						categgory_id =int(row[0])
						scene_small[categgory_id] += 1

	# print len(scene_small)

	# for index, value in enumerate(scene_small):
	# 	print index, value, small_category[index]

	data_205, data_28 = writeDataToClass(scene_vec, category,scene_small, small_category)
	writeToJson(data_205, data_28, movie_id)
	movie_counter += 1



