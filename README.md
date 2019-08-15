# Deep learning with Tensorflow
Constructing, training and testing deep neural networks in python using Tensorflow 

Inspired by 'Neural Networks and Deep Learning' by Michael Nielsen. In chapter 6 of the book, construction of deep network layer by layer is demonstrated using Theano. In this project, a similar approach is followed to construct netwrok architectures using Tensorflow. 

Three kinds of layers can be called to build a given network architecture: fully connected, convolutional and softmax. Note that the convolutional layer already includes subsampling based on max-pooling. 

The file build_network.py is the one that constructs, trains and test the architecture from a given list of layers and training parameters. The list of layers and training parameters are provided to it usng anothe file called run_test_net.py. All that is needed to build a network and run a test is the 'run_test_net.py' file, which imports all the rest. Sample 2D image recognition test included in it using the MNIST database. 
