import json
import os

'''
this script takes in a batch of scene json files and assign tags to the ones that has a dominant scene catogory, here is categories that both ranked top3 and over 5% of the total farmes
'''

def getListOfFiles(source_folder):
	file_name_list = []
	for root, dirs, files in os.walk(source_folder):
		return files
		



scene_json_folder = '/home/vionlabs/Documents/scene_classification/Json/json_28'
output_path = '/home/vionlabs/Documents/scene_classification/Json/boxer_scene_output.json'
json_list = getListOfFiles(scene_json_folder)
print json_list
category_filter_list = ['likely_a_face', 'outdoor', 'indoor']
json_output = open(output_path, 'w')
for imdbId in json_list:
	if os.path.isfile(scene_json_folder + '/' + imdbId):
		scene_reader = open(scene_json_folder + '/' + imdbId)
		for line in scene_reader:
			scene_freq = json.loads(line)
			scene_response = {'imdb_id': scene_freq['movie id']}
			scene_response['scene_tag'] = []
			for json_categories in reversed(scene_freq['environment']):
				category = json_categories.keys()
				frame_counter = json_categories.values()
				#print frame_counter[0]
				if frame_counter[0] > 0.15 * scene_freq['scene frame number']:
					if category[0] not in category_filter_list:
						print json_categories
						scene_tag = {category[0]: float(frame_counter[0])/scene_freq['scene frame number']}
						print scene_tag
						scene_response['scene_tag'].append(scene_tag)
						print scene_response['scene_tag']
				else:
					break
			
			json.dump(scene_response, json_output)
			json_output.write('\n')

json_output.close()

