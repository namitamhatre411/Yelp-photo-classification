import numpy as np
import scipy.io
import cPickle as pickle

arr = scipy.io.loadmat('fv_unnorm_test.mat');
arr = arr['fv_unnorm_test'];

with open('../KaggleData/data_pickle/images_train.pickle','rb') as handle:
    images_train = pickle.load(handle)
    
with open('../KaggleData/data_pickle/bid_to_image.pickle', 'rb') as handle:
    bid_to_image = pickle.load(handle)

with open('../KaggleData/data_pickle/image_to_bid.pickle', 'rb') as handle:
    image_to_bid = pickle.load(handle)

with open('../KaggleData/data_pickle/bid_to_label.pickle', 'rb') as handle:
    bid_to_label = pickle.load(handle)

with open('../KaggleData/data_pickle/bid_train.pickle', 'rb') as handle:
    bid_train = pickle.load(handle)

with open('../KaggleData/data_pickle/bid_test.pickle', 'rb') as handle:
    bid_test = pickle.load(handle)

with open('../KaggleData/data_pickle/images_train.pickle', 'rb') as handle:
    images_train = pickle.load(handle)

with open('../KaggleData/data_pickle/images_test.pickle', 'rb') as handle:
    images_test = pickle.load(handle)

arr = np.array(arr)
arr = arr.reshape(len(arr),len(arr[0][0]))

Ytest = [];

for i in images_test:
	Ytest.append(bid_to_label[image_to_bid[i]])


#Shuffle

import sklearn
arr, Ytest = sklearn.utils.shuffle(arr, Ytest, random_state=0)

from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC

pred = clf.predict(arr)

with open('predictions.pickle', 'wb') as handle:
	pickle.dump(pred, handle)

with open('ytest.pickle', 'wb') as handle:
	pickle.dump(Ytest, handle)
