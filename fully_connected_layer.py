# -*- coding: utf-8 -*-
"""
Created on Thu Aug 15 09:57:29 2019

@author: Surafel W.
"""

from __future__ import absolute_import, division, print_function
import tensorflow as tf
relu = tf.nn.relu


###############################################################################
class FullyConnectedLayer(object):
    
    def __init__(self, n_in, n_out, activation_fn = relu, 
                 keep_prob = 1.0, param_initialization = 'tf_truncated_norm'):
        
        self.n_in = n_in
        self.n_out = n_out
        self.keep_prob = keep_prob
        self.activation_fn = activation_fn
        self.param_initialization = param_initialization
        
        
        self.b = tf.Variable(tf.constant(0.1, shape = [self.n_out]))
        
        if self.param_initialization == 'tf_truncated_norm':
            # default
            self.w = tf.Variable(tf.truncated_normal([self.n_in, self.n_out], stddev = 0.1))
        
        elif self.param_initialization == 'const_scale':
            # option_2
            self.w = tf.random.normal([self.n_in, self.n_out], stddev = (1. / self.n_in))
        self.params = [self.w, self.b]
        
    
    def layer_input_output(self, l_input):
        
        ''' 
        Set layer input (Flattening input from conv_layer)
        
        If input is alaready flattened, shape (mini_batch_size, num_input_nodes),
        the flattening through reshaping has no effect
        '''
        self.l_input = tf.reshape(l_input, [-1, self.n_in])
        
        # Form the fully connected layer before dropout
        self.fc_out = tf.add(tf.matmul(self.l_input, self.w), self.b)

        # Apply ReLU (or other provided activation layer)
        activated_fc = self.activation_fn(self.fc_out)
        
        # layer output with or without dropout depending on keep_prob provided
        self.l_output = tf.nn.dropout(activated_fc, self.keep_prob)
        
        
#'''
#Testing input and ouput formats of FullyConnectedLayer ...
#'''
#test_in = tf.placeholder(tf.float32, shape = [None, 784])
#test_out = tf.placeholder(tf.float32, shape = [None, 10]) 
#
#n_in = 784
#n_out = 40
#
#fc_layer_test = FullyConnectedLayer(n_in, n_out)
#fc_layer_test.layer_input_output(test_in)
#fc_layer_test.keep_prob
#fc_layer_test.l_output
