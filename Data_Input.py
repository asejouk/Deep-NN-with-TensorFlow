#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 18 14:05:54 2017

@author: asejouk
"""




import L_layer_model_tf


import time
import numpy as np
import h5py
import matplotlib.pyplot as plt
import scipy
import scipy.io as sio
# Load data

def convert_to_one_hot(Y, C):
    Y = np.eye(C)[Y.reshape(-1)].T
    return Y


def load_dataset():
    train_dataset = h5py.File('data/train_signs.h5', "r")
    train_set_x_orig = np.array(train_dataset["train_set_x"][:]) # your train set features
    train_set_y_orig = np.array(train_dataset["train_set_y"][:]) # your train set labels

    test_dataset = h5py.File('data/test_signs.h5', "r")
    test_set_x_orig = np.array(test_dataset["test_set_x"][:]) # your test set features
    test_set_y_orig = np.array(test_dataset["test_set_y"][:]) # your test set labels

    classes = np.array(test_dataset["list_classes"][:]) # the list of classes

    train_set_y_orig = train_set_y_orig.reshape((1, train_set_y_orig.shape[0]))
    test_set_y_orig = test_set_y_orig.reshape((1, test_set_y_orig.shape[0]))

    return train_set_x_orig, train_set_y_orig, test_set_x_orig, test_set_y_orig, classes



X_train_orig, Y_train_orig, X_test_orig, Y_test_orig, classes = load_dataset()

#Data=sio.loadmat('Data.mat')
#X_train_orig=Data['X_train'];
#Y_train_orig=Data['Y_train'];
#Y_dev_orig=Data['Y_dev'];
#X_dev_orig=Data['X_dev'];

# Flatten the training and test images
X_train_flatten = X_train_orig.reshape(X_train_orig.shape[0], -1).T
X_dev_flatten = X_dev_orig.reshape(X_dev_orig.shape[0], -1).T

# Normalize image vectors
X_train = X_train_flatten
X_dev = X_dev_flatten

# Convert training and test labels to one hot matrices
Y_train = convert_to_one_hot(Y_train_orig, 17)
Y_dev = convert_to_one_hot(Y_dev_orig, 17)

layers_dims=[6006,25,12,17]

#parameters = model(X_train, Y_train, X_test, Y_test,layers_dims)

learning_rate = 0.0001
number_iter = 1500
minibatch_size = 32
lambd = 3
keep_prob =0.7
print_cost = True
model = None



parameters=tf_L_layer_model_with_Adam(X_train, Y_train, X_dev, Y_dev,layers_dims, learning_rate, number_iter, minibatch_size, print_cost)






