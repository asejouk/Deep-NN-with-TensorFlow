#!/usr/bin/env python3
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
from pdb import set_trace as bp



x=tf.placeholder(dtype=tf.int64,name="x")

a=tf.constant(10,dtype=tf.int64,name="a")
b=tf.constant(10,dtype=tf.int64,name="b")
z=tf.Variable(a*b,name="z")

y=x*2



init=tf.global_variables_initializer()

with tf.Session() as session:
    session.run(init)
    print(session.run(y*z,feed_dict={x:3}))
    
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
    #with tf.variable_scope(name=None) as scope:
        #scope.reuse_variables()
    for i in range(L-1):
        #parameters["W"+str(i+1)] = tf.get_variable("W1",[25,12288],initializer=tf.contrib.layers.xavier_initializer(seed=1))
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


def model(X_train, Y_train, X_test, Y_test,layers_dims, learning_rate = 0.0001,
          num_epochs = 1500, minibatch_size = 32, print_cost = True):

    tf.reset_default_graph()                          # to be able to rerun the model without overwriting tf variables
    tf.set_random_seed(1)                             # to keep consistent results
    seed = 3                                          # to keep consistent results
    (n_x, m) = X_train.shape                          # (n_x: input size, m : number of examples in the train set)
    n_y = Y_train.shape[0]                            # n_y : output size
    costs = []                                        # To keep track of the cost
    
    # Create Placeholders of shape (n_x, n_y)
    ### START CODE HERE ### (1 line)
    X, Y = create_placeholders(n_x,n_y)
    ### END CODE HERE ###

    # Initialize parameters
    ### START CODE HERE ### (1 line)
    parameters = initialize_parameters(layers_dims)
    ### END CODE HERE ###
    
    # Forward propagation: Build the forward propagation in the tensorflow graph
    ### START CODE HERE ### (1 line)
    Z3,_ = forward_propagation(X,parameters)
    ### END CODE HERE ###
    
    # Cost function: Add cost function to tensorflow graph
    ### START CODE HERE ### (1 line)
    cost = compute_cost(Z3,Y)
    ### END CODE HERE ###
    
    # Backpropagation: Define the tensorflow optimizer. Use an AdamOptimizer.
    ### START CODE HERE ### (1 line)
    optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate).minimize(cost)
    ### END CODE HERE ###
    
    # Initialize all the variables
    init = tf.global_variables_initializer()

    # Start the session to compute the tensorflow graph
    with tf.Session() as sess:
        
        # Run the initialization
        sess.run(init)
        
        # Do the training loop
        for epoch in range(num_epochs):

            epoch_cost = 0.                       # Defines a cost related to an epoch
            num_minibatches = int(m / minibatch_size) # number of minibatches of size minibatch_size in the train set
            seed = seed + 1
            minibatches = random_mini_batches(X_train, Y_train, minibatch_size, seed)

            for minibatch in minibatches:

                # Select a minibatch
                (minibatch_X, minibatch_Y) = minibatch
                
                # IMPORTANT: The line that runs the graph on a minibatch.
                # Run the session to execute the "optimizer" and the "cost", the feedict should contain a minibatch for (X,Y).
                ### START CODE HERE ### (1 line)
                _ , minibatch_cost = sess.run([optimizer, cost], feed_dict={X: minibatch_X, Y: minibatch_Y})
                ### END CODE HERE ###
                
                epoch_cost += minibatch_cost / num_minibatches

            # Print the cost every epoch
            if print_cost == True and epoch % 100 == 0:
                print ("Cost after epoch %i: %f" % (epoch, epoch_cost))
            if print_cost == True and epoch % 5 == 0:
                costs.append(epoch_cost)
                
        # plot the cost
        plt.plot(np.squeeze(costs))
        plt.ylabel('cost')
        plt.xlabel('iterations (per tens)')
        plt.title("Learning rate =" + str(learning_rate))
        plt.show()

        # lets save the parameters in a variable
        parameters = sess.run(parameters)
        print ("Parameters have been trained!")

        # Calculate the correct predictions
        correct_prediction = tf.equal(tf.argmax(Z3), tf.argmax(Y))

        # Calculate accuracy on the test set
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

        print ("Train Accuracy:", accuracy.eval({X: X_train, Y: Y_train}))
        print ("Test Accuracy:", accuracy.eval({X: X_test, Y: Y_test}))
        
        return parameters




#with tf.Session() as sess:
#    X, Y = create_placeholders(12288, 6)
#    parameters = initialize_parameters(layers_dims)
#    Z3,_ = forward_propagation(X, parameters)
#    cost = compute_cost(Z3, Y)
#    print("cost = " + str(cost))


    

    
