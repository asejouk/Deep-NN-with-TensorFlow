
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 27 13:36:49 2017

@author: asejouk
"""

import math
import numpy as np
import h5py
import matplotlib.pyplot as plt
import tensorflow as tf
#from pdb import set_trace as bp





def sigmoid(z):

    x=tf.placeholder(tf.float32,name="x")
    sigmoid=tf.sigmoid(x)
    with tf.Session() as sess:
        result=sess.run(sigmoid,feed_dict={x:z})

    return result

def cost(logits,labels):

    x=tf.placeholder(tf.float32,name="x")
    y=tf.placeholder(tf.float32,name="y")

    cost=tf.nn.sigmoid_cross_entropy_with_logits(logits=x,labels=y)

    with tf.Session() as session:
        result=session.run(cost,feed_dict={x:logits,y:labels})

    return result

def random_mini_batches(X, Y, mini_batch_size = 64, seed = 0):


    m = X.shape[1]                  # number of training examples
    mini_batches = []
    np.random.seed(seed)

    # Step 1: Shuffle (X, Y)
    permutation = list(np.random.permutation(m))
    shuffled_X = X[:, permutation]
    shuffled_Y = Y[:, permutation].reshape((Y.shape[0],m))

    # Step 2: Partition (shuffled_X, shuffled_Y). Minus the end case.
    num_complete_minibatches = math.floor(m/mini_batch_size) # number of mini batches of size mini_batch_size in your partitionning
    for k in range(0, num_complete_minibatches):
        mini_batch_X = shuffled_X[:, k * mini_batch_size : k * mini_batch_size + mini_batch_size]
        mini_batch_Y = shuffled_Y[:, k * mini_batch_size : k * mini_batch_size + mini_batch_size]
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)

    # Handling the end case (last mini-batch < mini_batch_size)
    if m % mini_batch_size != 0:
        mini_batch_X = shuffled_X[:, num_complete_minibatches * mini_batch_size : m]
        mini_batch_Y = shuffled_Y[:, num_complete_minibatches * mini_batch_size : m]
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)

    return mini_batches


def create_placeholders(n_x,n_y):

    x=tf.placeholder(tf.float32,shape=[n_x, None],name="x")

    y=tf.placeholder(tf.float32,shape=[n_y,None],name="y")

    return x,y

def initialize_parameters(layers_dims):

    tf.set_random_seed(1)
    L=len(layers_dims)
    parameters={}

    for i in range(L-1):
        parameters["W"+str(i+1)] = tf.get_variable("W"+str(i+1),[layers_dims[i+1],layers_dims[i]],initializer=tf.contrib.layers.xavier_initializer(seed=1))
        parameters["b"+str(i+1)] = tf.get_variable("b"+str(i+1),[layers_dims[i+1],1],initializer=tf.zeros_initializer())

    return parameters

def forward_propagation(X,parameters):

    L=len(parameters)//2
    A=X
    for i in range (L):
        Z = tf.add(tf.matmul(parameters["W"+str(i+1)],A),parameters["b"+str(i+1)])                                              # Z1 = np.dot(W1, X) + b1
        A = tf.nn.relu(Z)
    return Z,A

def compute_cost(Z3,Y):

    logits = tf.transpose(Z3)
    labels = tf.transpose(Y)

    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits,labels=labels))

    return cost


