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

tf.import_graph_def(graph_def)

with tf.Session() as sess:
    for operation in sess.graph.get_operations():
        print(operation.name)
        for k in operation.inputs:
            print operation.name,"Input ",k.name,k.get_shape()
        for k in operation.outputs:
            print operation.name,"Output ",k.name
        print "\n"