# Implementation-of-neural-network-on-MNIST-data


**Please note that the explanation and results are all tabulated in the jupyter notebook file Neural_network_our_implementation.ipynb.
Hence, we did not explicity include any document**

Dataset: ex3data1.mat- We use a dataset which is a part of MNIST datset. This contains 5000 handwritten instances. Each of these is a
grayscale 20X20 image. The labels contains 10 classes each of which represents digit from 0 to 9.

Files:

Neural_network_our_implementation.ipynb : We have used jupyter notebook to construct the neural network. This file contains all the
necessary explanation and justifications. All the results and computations are displayed and can be easily understood. This is our 
implementation of Neural Networks on the subset of MNIST dataset. We obtained accuracy of 93.1% without KFold and 93% with Kfold cross
validation. 

Neural_Network_using_sklearn.py: This is neural network implementation using Sklearn library on the same dataset. All the parameters like input size, hidden layer( which is one) and hidden nodes are same as in our implementation. Using 5 fold cross validation, we have obtained an accuracy of 91.9%. This is a python file which can be executed on linux terminal using 
'python Neural_Network_using_sklearn.py' command.

sklearn_accuracy.png: Screenshot which contains accuracy of the implementation using sklearn library






