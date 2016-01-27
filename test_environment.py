from classification_revised import EnvironmentClassifier
from vionmodels.research import Environments401

def runtime():
	#path for all the files needed
	MODEL_FILE = '/home/vionlabs/Documents/scene_classification/places_models/vgg16_places2/deploy.prototxt'
	PRETRAINED = '/home/vionlabs/Documents/scene_classification/places_models/vgg16_places2/vgg16_places2.caffemodel'
	VIDEOPATH = '../video_file/welcome_to_sweden.mp4'
	MEAN_FILE = '../places_models/Places_CNDS_model/places_mean.npy'
	category_path = '/home/vionlabs/Documents/scene_classification/places_models/vgg16_places2/categories.txt'

	EC = EnvironmentClassifier()
	EC.get_category_list(category_path)
	EC.deploy_network(MODEL_FILE, PRETRAINED, MEAN_FILE)
	EC.video_reader(VIDEOPATH)
	env_data = EC.environment_classification()
	env_timestamp = EC.set_timestamp()
	print env_data
	print env_timestamp

	# return Environments401(data = env_data, timestamp = env_timestamp)


if __name__ == "__main__":
	runtime()
