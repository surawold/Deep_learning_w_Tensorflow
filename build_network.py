# -*- coding: utf-8 -*-
"""
Created on Thu Aug 15 10:06:19 2019

@author: Surafel W.
"""

from __future__ import absolute_import, division, print_function
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
relu = tf.nn.relu

###############################################################################
class network(object):
    
    def __init__(self, layers, mini_batch_size,
                 input_size = 784,
                 output_size = 10):
        ''' For a given network architecture and mb_size
        trains the network parameters using SGD
        
        When calling this object, sample line:
            
            net = Network([
                    ConvPoolLayer(image_shape=(mini_batch_size, 1, 28, 28), 
                      filter_shape=(20, 1, 5, 5), 
                      poolsize=(2, 2), 
                      activation_fn=ReLU),
                    
                    ConvPoolLayer(image_shape=(mini_batch_size, 20, 12, 12), 
                      filter_shape=(40, 20, 5, 5), 
                      poolsize=(2, 2), 
                      activation_fn=ReLU),
                    
                    FullyConnectedLayer(n_in=40*4*4, n_out=100, activation_fn=ReLU),
                    FullyConnectedLayer(n_in=100, n_out=100, activation_fn=ReLU),
                    SoftmaxLayer(n_in=100, n_out=10)], mini_batch_size)
                    
            
            NOTE:
                When determining image_shape (output from prev layer), 
                depending on what kind of padding (VALID or SAME), 
                the image_shape will need to be determined accurately
                according to the convolution operation in the prev_layer
        
        '''
        self.input_size = input_size
        self.output_size = output_size
        
        self.layers = layers
        self.mini_batch_size = mini_batch_size
        self.params = [param for layer in layers for param in layer.params]
        
        
        ''' Net_input:
            For mnist, inputs are in a vector (784, ) shape
            So input_shape = [mini_batch_size, 784]
            mini_batch_size left as None:
                automatically detected when feeding an input batch during training '''
        self.x = tf.placeholder(tf.float32, shape = [None, self.input_size])
        
        
        ''' Desired_output:
            10 different classes --> shape = [batch_size, 10]'''
        self.y = tf.placeholder(tf.float32, shape = [None, self.output_size])
        
        ''' Dropout keep_probability:
            Anytime dropout is used, during training steps, value used
            
            The default value of keep_prob for each layer (1.0) replaced by this
        '''
        self.keep_prob = tf.placeholder(tf.float32)
        for layer in self.layers:
            layer.keep_prob = self.keep_prob
        
        
        
        '''
        Activate the input and output of each layer provided as input
        '''
        init_layer = self.layers[0]
        init_layer.layer_input_output(self.x)
        
        for j in range(1, len(self.layers)):
            prev_layer, layer = self.layers[j-1], self.layers[j]
            layer.layer_input_output(prev_layer.l_output)
        
        self.output = self.layers[-1].l_output
    
    
    
    
    def generate_mini_batches(self, dataset):
        
        mb_size = self.mini_batch_size
        np.random.shuffle(dataset)
        mini_batches = []
        idx_max = int(dataset.shape[0] / mb_size) 
        
        for i in range(idx_max):
            mini_batches.append([dataset[i*mb_size:(i+1)*mb_size, :-10], dataset[i*mb_size:(i+1)*mb_size, -10:]])
    
        return mini_batches




    def SGD(self, training_data, epochs, keep_prob,
            mini_batch_size, validation_data, test_data,
            learn_rate_initial = 1e-4, cost_func = None, train_method = None,
            global_step = tf.Variable(0, trainable=False),
            classification_labels = np.arange(10)):
        '''
        Stochastic Gradient Descent Learning In Progress...
        
        '''
        
        self.learn_rate_initial = learn_rate_initial
        self.global_step = global_step
        
        # Define a cost function (Default: Cross-Entropy)
        # This can later be modified to handle a given cost function
        if not cost_func:
            cost_fun = tf.reduce_mean(-tf.reduce_sum(self.y * tf.log(self.output), reduction_indices = [1]))
        else:
            cost_fun = cost_fun
         
            
        
        # Define an Optimizer (Default: AdamOptimizer)
        # Other options handled
        if not train_method:
            # When train_method left as None, default method used
            train_method = tf.train.AdamOptimizer(self.learn_rate_initial).minimize(cost_fun)
        
        else:
            if train_method == 'GD_exp_dec_lr':
                '''
                Exponential decay learning rate:
                    Formula:
                        lr[i] = lr[i-1] * decay_rate ^ (global_step / decay_steps)
                    Call:
                        tf.train.exponential_decay(
                        learning_rate,
                        global_step,
                        decay_steps,
                        decay_rate,
                        staircase=False,
                        name=None)
                '''
                
                self.learning_rate = tf.train.exponential_decay(learn_rate_initial, self.global_step, 
                                                           1000000, 0.95, staircase=True)
                train_method = tf.train.GradientDescentOptimizer(self.learning_rate).minimize(cost_fun, self.global_step)
            else:
                '''
                If providing other optimization type:
                    call SGD with:
                        train_method = tf.train.GradientDescentOptimizer(learning_rate)
                '''
                train_method = train_method.minimize(cost_fun)
        
        
        # Define function showing the correct prediction for a given batch
        correct_prediction = tf.equal(tf.argmax(self.output, 1), tf.argmax(self.y, 1))
        
        # Define function evaluating correct number of prediction for a given batch
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        
        
        '''
        Run session - Begin Training ...
        '''
        sess_net = tf.InteractiveSession()
        sess_net.run(tf.global_variables_initializer())
        
        label_predicted = tf.argmax(self.output, 1)
        label_true = tf.arg_max(self.y, 1)
        
        self.training_accuracy = []
        self.validation_accuracy = []
        self.test_accuracy = []
        
        for epoch in range(epochs):
            
            
            '''
            Generate mini-batches:
                dataset = np.concatenate((features, labels), axis = 1):
                    so that shuffling can be done every time mini-batch generated
                    
                    In the case of of mnist - dataset.shape = (55000, 794):
                        (55000, 784) - images
                        (55000, 10) - labels
            
                return = [mini-batches]:
                    length = len(dataset) / mb_size
                    Each element of this list has two elements with:
                        shape (mini_batch_size, 784) - images
                        shape (mini_batch_size, 10) - labels
            '''
        
            dataset = np.concatenate((training_data[0], training_data[1]), axis = 1)
            
            mini_batches = self.generate_mini_batches(dataset)
            num_steps_per_epoch = len(mini_batches)
            
            # Update the initial learning rate used in AdamOptimizer in every epoch
            self.learn_rate_initial = self.learn_rate_initial / (1 + (epoch * num_steps_per_epoch))
            
            for i in range(num_steps_per_epoch):
                batch = mini_batches[i]
                if i % 1100 == 0:
                    
                    training_accuracy = accuracy.eval(feed_dict = {self.x: training_data[0], 
                                                                   self.y: training_data[1], 
                                                                   self.keep_prob: 1.0})
                    self.training_accuracy.append(training_accuracy * 100) # [%]
                    
                    validation_accuracy = accuracy.eval(feed_dict = {self.x: validation_data[0], 
                                                                     self.y: validation_data[1], 
                                                                     self.keep_prob: 1.0})
                    print("Epoch {0}, Step {1}: validation accuracy {2:.2%}".format(epoch, i, validation_accuracy))
                    self.validation_accuracy.append(validation_accuracy * 100) # [%]
                
                    test_accuracy = accuracy.eval(feed_dict = {self.x: test_data[0], 
                                                               self.y: test_data[1], 
                                                               self.keep_prob: 1.0})
                    print("Corrsponding test accuracy {0:.2%}".format(test_accuracy))
                    self.test_accuracy.append(test_accuracy * 100) # [%]
                
                train_method.run(feed_dict = {self.x: batch[0], self.y: batch[1], self.keep_prob: keep_prob})
        
        
        # At the end of training, show the ConfusionMatrix on the test dataset
        predicted_label = label_predicted.eval(feed_dict = {self.x: test_data[0],
                                                            self.y: test_data[1],
                                                            self.keep_prob: 1.0})
        true_label = label_true.eval(feed_dict = {self.x: test_data[0],
                                                  self.y: test_data[1],
                                                  self.keep_prob: 1.0})
        
        self.confusion_matrix = confusion_matrix(true_label, predicted_label)
        fig, ax = plt.subplots()
        im = ax.imshow(self.confusion_matrix,
                       interpolation = 'nearest',
                       cmap = 'Blues')
        ax.figure.colorbar(im, ax = ax)
        ax.set(xticks = np.arange(self.confusion_matrix.shape[1]),
               yticks = np.arange(self.confusion_matrix.shape[0]),
               xticklabels = classification_labels,
               yticklabels = classification_labels,
               ylabel = 'True label',
               xlabel = 'Predicted label')
        fmt = 'd'
        thresh = 0.1 * self.confusion_matrix.max()
        for i in range(self.confusion_matrix.shape[0]):
            for j in range(self.confusion_matrix.shape[1]):
                ax.text(j, i, format(self.confusion_matrix[i, j], fmt),
                        ha = "center", va = "center",
                        color = "white" if self.confusion_matrix[i, j] > thresh else "black")
        fig.tight_layout()
        
        # Plot random samples with the corresponding true and predicted labels 
        rand_idx = np.random.randint(0, len(test_data[1]), 15)
        plt.figure(figsize=(20, 20))
        n_rows = 3
        n_columns = 5
        sample_images = test_data[0][rand_idx]
        for i in range(len(rand_idx)):
            plt.subplot(n_rows, n_columns, i+1)
            sample_pred_label = classification_labels[predicted_label[rand_idx[i]]]
            sample_true_label = classification_labels[true_label[rand_idx[i]]]
            plt.imshow(sample_images[i].reshape(28, 28), cmap = 'gray')
            plt.title('Label predicted: ' + str(sample_pred_label) + '\nLabel true: ' + str(sample_true_label))
        
    
    
    def plot_layer_output(self, image_input, layer_number, 
                          num_features = 15, Conv1D = False):
        
        sess_net = tf.InteractiveSession()
        sess_net.run(tf.global_variables_initializer())
        '''
        Set the given layer parameters as the trained parameters
        '''
        self.layers[layer_number].W = sess_net.run(self.params[layer_number][0])
        self.layers[layer_number].b = sess_net.run(self.params[layer_number][1])
            
        layer_output = sess_net.run(self.layers[layer_number].l_output,
                                feed_dict = {self.x: image_input, self.keep_prob: 1.0})
        
        filter_idx = np.arange(layer_output.shape[-1])
        np.random.shuffle(filter_idx)
        
        plt.figure(figsize=(20, 20))
        n_rows = 3
        n_columns = num_features / n_rows 
        
        if Conv1D:
            for i in range(num_features):
                plt.subplot(n_rows, n_columns, i+1)
                plt.title('Filter ' + str(i))
                plt.plot(layer_output[0,:,:,i])
        else:
            for i in range(num_features):
                plt.subplot(n_rows, n_columns, i+1)
                plt.title('Filter ' + str(i))
                plt.imshow(layer_output[0,:,:,i], interpolation="nearest", cmap="gray")
                
