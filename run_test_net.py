# -*- coding: utf-8 -*-
"""
Created on Thu Aug 15 10:11:47 2019

@author: Surafel W.
"""

from __future__ import absolute_import, division, print_function
import matplotlib.pyplot as plt
import tensorflow as tf
relu = tf.nn.relu
import time


''' Import modules with layers that makeup deep-net '''
import fully_connected_layer
import convolutional_layer
import softmax_layer
import build_network

FullyConnectedLayer = fully_connected_layer.FullyConnectedLayer
ConvPoolLayer = convolutional_layer.ConvPoolLayer
SoftmaxLayer = softmax_layer.SoftmaxLayer
BuildDeepNet = build_network.network


''' Import datasets - MNIST '''
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot = True)

training_images, training_labels = mnist.train.images, mnist.train.labels
#training_images.shape, training_labels.shape
training_data = (training_images, training_labels)


validation_images, validation_labels = mnist.validation.images, mnist.validation.labels 
#validation_images.shape, validation_labels.shape
validation_data = (validation_images, validation_labels)


test_images, test_labels = mnist.test.images, mnist.test.labels
#test_images.shape, test_labels.shape
test_data = (test_images, test_labels)

        

net_layers = [
        ConvPoolLayer([5, 5, 1, 10], [-1, 28, 28, 1]),
        ConvPoolLayer([5, 5, 10, 20], [-1, 14, 14, 10]),
        FullyConnectedLayer(7*7*20, 100),
        SoftmaxLayer(100, 10)
        ]

sess_net = tf.InteractiveSession()
#print(sess_net._closed)
sess_net.close()

mini_batch_size = 10
deep_net_mnist = BuildDeepNet(net_layers, mini_batch_size)     

epochs = 3
keep_prob = 0.5

start_time = time.time()
deep_net_mnist.SGD(training_data, epochs, keep_prob,
                   mini_batch_size, validation_data, test_data,
                   learn_rate_initial = 1e-4)
elapsed_time = time.time() - start_time
print("elapsed_time: {0:.2} minutes".format(elapsed_time / 60.))

# Plot intermediate features
image_sample = training_images[350]
deep_net_mnist.plot_layer_output(image_sample.reshape(1, 784), 0)
deep_net_mnist.plot_layer_output(image_sample.reshape(1, 784), 1)

# Plot the training and validation accuracies
plt.figure()
plt.plot(deep_net_mnist.training_accuracy, 'C1', label = 'Training')
plt.plot(deep_net_mnist.validation_accuracy, 'C2', label = 'Validation')
plt.xlabel('Number of learning steps [mini-batches]')
plt.ylabel('Classification accuracy [%]')
plt.legend()



