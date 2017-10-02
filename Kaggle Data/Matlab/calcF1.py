import numpy as np
import scipy.io
import cPickle as pickle

with open('ytest.pickle', 'rb') as handle:
	Ytest = pickle.load(handle)

with open('predictions.pickle', 'rb') as handle:
	pred = pickle.load(handle)

#Precision = 
F1num = 0
F1den = 0

from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC

Ytest = MultiLabelBinarizer().fit_transform(Ytest)

for i in range(9):
	tp=0.0
	all_pos=0.0
	all_pred_pos=0.0

	for j in range(len(Ytest)):
		if(Ytest[j][i]==1 and pred[j][i]==1):
			tp+=1
		if(Ytest[j][i]==1):
			all_pos+=1
		if(pred[j][i]==1):
			all_pred_pos+=1

	precision = tp/all_pos
	recall = tp/all_pred_pos

	F1num += 2*precision*recall*all_pos/(precision+recall)
	F1den += all_pos

print('Mean F1 score is')
print(F1num/F1den)