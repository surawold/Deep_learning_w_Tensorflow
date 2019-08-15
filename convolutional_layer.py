# -*- coding: utf-8 -*-
"""
Created on Thu Aug 15 09:57:30 2019

@author: Surafel W.
"""

from __future__ import absolute_import, division, print_function
import tensorflow as tf
relu = tf.nn.relu

###############################################################################
class ConvPoolLayer(object):
    
    '''
    Combininig convolution and max-pooling layers together (simplifies code):
        Given a batch of images (features from prev_layer):
            apply convolution using filter_shape
            apply max_pooling (or other pooling) to this result
            apply relu (or other activation_function) to this result
        Output:
            the resulting features with and withoug dropout
    
    
    Description of inputs:
        filter_shape:
            [filter_height, filter_width, in_channels, out_channels]
        image_shape (input_feature_shape):
            [batch, in_height, in_width, in_channels]
        padding: 
            A string from: "SAME", "VALID"
        strides:
            A list of ints. 1-D tensor of length 4. 
            The stride of the sliding window for each dimension of input. 
            
    '''
    
    def __init__(self, filter_shape, image_shape,
                 poolsize = (2, 2), activation_fn = relu, keep_prob = 1.0,
                 strides_conv = [1, 1, 1, 1], strides_pool = [1, 2, 2, 1],
                 padding = 'SAME', param_initialization = 'tf_truncated_norm',
                 ws_init = None, w_init_given = False):
        
        self.filter_shape = filter_shape
        self.image_shape = image_shape
        self.poolsize = poolsize 
        self.activation_fn = activation_fn
        self.strides_conv = strides_conv
        self.strides_pool = strides_pool
        self.padding = padding
        self.param_initialization = param_initialization
        
        self.keep_prob = keep_prob
        
        n_in = self.filter_shape[2]
        
        ''' 
        *** Weight initialization makes all the difference ***
                
                In order to avoid network parameter instability 
                (due to saturation of the activation function), 
                two methods explored:
                    
                    (1) tensorflow_truncated_norm (default):
                        cuts of random numbers outsided 2 * (stddev_given)
                        
                    (2) input nodes dependent standard deviation (Michael Nielsen ch3):
                        to keep the scale similar in all layers 
                        stddev = np.sqrt(1/ n_in)
                        
                        NOTE:
                            Google search revealed that this method
                            goes by the name * Xavier_initializer *
        
        '''
        
        self.b = tf.Variable(tf.constant(0.1, shape = [self.filter_shape[-1]]))

        if w_init_given:
            
            self.W = ws_init

        else:
            
            if self.param_initialization == 'tf_truncated_norm':
                # default
                self.W = tf.Variable(tf.truncated_normal(self.filter_shape, stddev = 0.1))
        
            elif self.param_initialization == 'const_scale':
                # option_2
                self.W = tf.random.normal(self.filter_shape, stddev = (1. / n_in))
        

        self.params = [self.W, self.b]
        
        
    def layer_input_output(self, layer_input):
        
        ''' 
        Reshaping output from ConvPoolLayer (if this is the case) has no effect
        since the output shape from this layer is already in the correct format
        ''' 
        self.l_input = tf.reshape(layer_input, self.image_shape)
        
        #Apply the filters and perfom convolution '''
        convolve = tf.nn.conv2d(self.l_input, self.W, 
                                strides = self.strides_conv, padding = self.padding) + self.b

        #ReLU (or other provided activation function) '''
        activated_conv = self.activation_fn(convolve)
        
        #Max pooling using the given poolsize and strides '''
        pooled_out = tf.nn.max_pool(activated_conv, ksize = [1, self.poolsize[0], self.poolsize[1], 1],
                                    strides = self.strides_pool, padding = self.padding)
        
        #Set the network output as the output from max_pullout
        self.l_output = pooled_out
        
    
#'''
#Testing input and ouput formats of ConvPoolLayer ...
#'''  
#test_in = tf.placeholder(tf.float32, shape = [None, 784])
#test_out = tf.placeholder(tf.float32, shape = [None, 10])      
#test_in
#test_out
#
#filter_shape = [28, 28, 1, 10]
#image_shape = [-1, 28, 28, 1]
#
#conv_layer_test = ConvPoolLayer(filter_shape, image_shape)
#conv_layer_test.layer_input_output(test_in)
#conv_layer_test.keep_prob
#conv_layer_test.l_output
#conv_layer_test.W
