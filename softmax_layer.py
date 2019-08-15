# -*- coding: utf-8 -*-
"""
Created on Thu Aug 15 09:57:32 2019

@author: Surafel W.
"""

from __future__ import absolute_import, division, print_function
import tensorflow as tf
relu = tf.nn.relu


###############################################################################
class SoftmaxLayer(object):
    
    def __init__(self, n_in, n_out,
                 keep_prob = 1.0, param_initialization = 'tf_truncated_norm'):
        
        '''
        For neuron j (layer l):
            input: x_in
            desired_output: y
            
            z_j = matrix_multiplication(x_in, weight) + bias
            a_j = activation_function(z_j)
        
        Softmax (activation_function):
            a_j = exp(z_j) / sum(exp(z_js))
            
        Log_liklihood of cost function:
            C = -ln(a_y), where y is the desired output (index)
        
        
        How Softmax + Loglikelyhood cost function works:
            Given an input (x_in) with a desired output value of y = 7:
                Network_output = 7:
                    a_7 ~ 1
                    C = -ln(a_7) ~ 0
                Network_output not = 7:
                    a_7 ~ 0
                    C = -ln(a_7) is large
                    
            This way, the desired properties of a cost function are achieved:
                always greater than zero, and
                as the decision gets worse, the cost gets larger
                
        '''
        
        self.n_in = n_in
        self.n_out = n_out
        self.keep_prob = keep_prob
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
        Set layer input (Flattened input from a fully connected layer)
        '''
        
        self.l_input = l_input
        
        # Same as creating a fully connected layer upto this point
        self.fc_out = tf.add(tf.matmul(self.l_input, self.w), self.b)
        
        # Apply Softmax
        self.softmax_out = tf.nn.softmax(self.fc_out)        
        
        # Apply dropout (if keep_prob different from default 1.0)
        # self.l_output = tf.nn.dropout(self.softmax_out, self.keep_prob)
        self.l_output = self.softmax_out


#'''
#Testing input and ouput SoftmaxLayer ...
#'''
#test_in = tf.placeholder(tf.float32, shape = [None, 784])
#test_out = tf.placeholder(tf.float32, shape = [None, 10]) 
#
#n_in = 784
#n_out = 10
#
#sm_layer_test = SoftmaxLayer(n_in, n_out)
#sm_layer_test.layer_input_output(test_in)
#sm_layer_test.keep_prob
#sm_layer_test.l_output
