import numpy as np
import scipy.io
import cPickle as pickle

with open('images_train.pickle','rb') as handle:
    images_train = pickle.load(handle)
    
with open('bid_to_image.pickle', 'rb') as handle:
    bid_to_image = pickle.load(handle)

with open('image_to_bid.pickle', 'rb') as handle:
    image_to_bid = pickle.load(handle)

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

Ytrain = [];

for i in images_train:
	Ytrain.append(bid_to_label[image_to_bid[i]])

from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC

Ytrain = MultiLabelBinarizer().fit_transform(Ytrain)

for i in range(9):
	all_pos=0.0
	for j in range(len(Ytrain)):
		if(Ytrain[j][i]==1):
			all_pos+=1
	print(all_pos)