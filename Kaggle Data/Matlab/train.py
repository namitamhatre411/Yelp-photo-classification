import numpy as np
import scipy.io
import cPickle as pickle

arr = scipy.io.loadmat('FV_unnorm.mat');
arr = arr['FV_unnorm'];

with open('../data_pickle/image_to_bid.pickle', 'rb') as handle:
	image_to_bid = pickle.load(handle)

with open('../data_pickle/bid_to_image.pickle', 'rb') as handle:
	bid_to_image = pickle.load(handle)

with open('../data_pickle/bid_to_label.pickle', 'rb') as handle:
	bid_to_label = pickle.load(handle)

with open('../data_pickle/bid_train.pickle', 'rb') as handle:
	bid_train = pickle.load(handle)

with open('../data_pickle/bid_test.pickle', 'rb') as handle:
	bid_test = pickle.load(handle)

with open('../data_pickle/images_train.pickle', 'rb') as handle:
	images_train = pickle.load(handle)

with open('../data_pickle/images_test.pickle', 'rb') as handle:
	images_test = pickle.load(handle)


arr = np.array(arr)
arr = arr.reshape(len(arr),len(arr[0][0]))

Ytrain = [];

for i in images_train:
	Ytrain.append(bid_to_label[image_to_bid[i]])


#Shuffle

import sklearn
arr, Ytrain = sklearn.utils.shuffle(arr, Ytrain, random_state=0)


# Train and Save

from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC

Ytrain = MultiLabelBinarizer().fit_transform(Ytrain)

clf = OneVsRestClassifier(SVC(kernel='linear'))
clf.fit(arr, Ytrain)

with open('VGG_Classifier.pickle', 'wb') as handle:
	pickle.dump(clf, handle)

pred = clf.predict(arr)

disp(pred)
