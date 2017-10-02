#https://github.com/janislejins/instantDreams/blob/master/ToDream/retraining.py

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio

model_fn = 'classify_image_graph_def.pb'

graph = tf.Graph()
sess = tf.InteractiveSession(graph=graph)

with tf.gfile.FastGFile(model_fn, 'rb') as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())
    
imagenet_mean = 117.0
tf.import_graph_def(graph_def)

import cPickle as pickle

with open('../KaggleData/data_pickle/images_train.pickle','rb') as handle:
    images_train = pickle.load(handle)
    
with open('../KaggleData/data_pickle/bid_to_image.pickle', 'rb') as handle:
    bid_to_image = pickle.load(handle)

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


with tf.Session() as sess:
#     for operation in sess.graph.get_operations():
#         print(operation.name)
#         for k in operation.inputs:
#             print operation.name,"Input ",k.name,k.get_shape()
#         for k in operation.outputs:
#             print operation.name,"Output ",k.name
#         print "\n"
        
#     softmax_tensor = sess.graph.get_tensor_by_name('import/softmax:0')

    representation_tensor = sess.graph.get_tensor_by_name('import/pool_3:0')

    print("Started computation for training set")
    
    for i in images_train:
        im1 = tf.gfile.FastGFile('../KaggleData/train_photos/'+str(i)+'.jpg', 'rb').read()    
        fv_unnorm = sess.run(representation_tensor, {'import/DecodeJpeg/contents:0': im1})
        fv_unnorm = fv_unnorm.reshape(1,2048);
        break
        
    count = 0
    for i in images_train:
    	print("Train :")
    	print(count)
        if(count==0):
            count+=1
            continue
        im1 = tf.gfile.FastGFile('../KaggleData/train_photos/'+str(i)+'.jpg', 'rb').read()    
        fv = sess.run(representation_tensor, {'import/DecodeJpeg/contents:0': im1})
        fv = fv.reshape(1,2048);
        fv_unnorm = np.append(fv_unnorm,fv,axis=0)
        count += 1

    sio.savemat('fv_unnorm.mat', {'fv_unnorm':fv_unnorm})

    print(fv_unnorm.shape)

    print("Started computation for test set")

    for i in images_test:
        im1 = tf.gfile.FastGFile('../KaggleData/train_photos/'+str(i)+'.jpg', 'rb').read()    
        fv_unnorm_test = sess.run(representation_tensor, {'import/DecodeJpeg/contents:0': im1})
        fv_unnorm_test = fv_unnorm_test.reshape(1,2048);
        break

    count = 0
    for i in images_test:
    	print("Test :")
    	print(count)
        if(count==0):
            count+=1
            continue
        im1 = tf.gfile.FastGFile('../KaggleData/train_photos/'+str(i)+'.jpg', 'rb').read()    
        fv = sess.run(representation_tensor, {'import/DecodeJpeg/contents:0': im1})
        fv = fv.reshape(1,2048);
        fv_unnorm_test = np.append(fv_unnorm_test,fv,axis=0)
        count +=1

    sio.savemat('fv_unnorm_test.mat', {'fv_unnorm_test':fv_unnorm_test})

    print(fv_unnorm_test.shape)

