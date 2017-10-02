import numpy as np
import scipy.io as sio
import cPickle as pickle


with open('image_to_bid.pickle', 'rb') as handle:
	image_to_bid = pickle.load(handle)

with open('bid_to_image.pickle', 'rb') as handle:
	bid_to_image = pickle.load(handle)

with open('bid_to_label.pickle', 'rb') as handle:
	bid_to_label = pickle.load(handle)

with open('bid_train.pickle', 'rb') as handle:
	bid_train = pickle.load(handle)

with open('bid_test.pickle', 'rb') as handle:
	bid_test = pickle.load(handle)

with open('images_train.pickle', 'rb') as handle:
	images_train = pickle.load(handle)

with open('images_test.pickle', 'rb') as handle:
	images_test = pickle.load(handle)


bid_train = np.array(bid_train);
bid_test = np.array(bid_test);
images_train = np.array(images_train);
images_test = np.array(images_test);


sio.savemat('image_to_bid.mat', {'image_to_bid': image_to_bid})
sio.savemat('bid_to_image.mat', {'bid_to_image': bid_to_image})
sio.savemat('bid_to_label.mat', {'bid_to_label': bid_to_label})
sio.savemat('bid_train.mat', {'bid_train': bid_train})
sio.savemat('bid_test.mat', {'bid_test': image_to_bid})
sio.savemat('images_train.mat', {'images_train': image_to_bid})
sio.savemat('images_test.mat', {'images_test': image_to_bid})

