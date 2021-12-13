# -*- coding: utf-8 -*-
"""
Created on Mon Dec  6 22:04:19 2021

@author: chira
"""

import numpy as np
from nn_utils import *


# Data
x = np.array([[0., 0., 0., 0.],
              [0., 0., 0., 1.],
              [0., 0., 1., 0.],
              [0., 0., 1., 1.],
              [0., 1., 0., 0.],
              [0., 1., 0., 1.],
              [0., 1., 1., 0.],
              [0., 1., 1., 1.],
              [1., 0., 0., 0.],
              [1., 0., 0., 1.],
              [1., 0., 1., 0.],
              [1., 0., 1., 1.],
              [1., 1., 0., 0.],
              [1., 1., 0., 1.],
              [1., 1., 1., 0.],
              [1., 1., 1., 1.]])

N = x.shape[0]

# Targets
y = (np.sum(x, axis=1) % 2).astype('int')


# Parameters
w1 = np.random.randn(4, 20)
b1 = np.zeros((20,))
w2 = np.random.randn(20, 2)
b2 = np.zeros((2,))


# NN deifnitions
def nn_forward(x, w1, b1, w2, b2):
    out_1, cache_1 = affine_forward(x, w1, b1)
    out_2, cache_2 = relu_forward(out_1)
    out_3, cache_3 = affine_forward(out_2, w2, b2)
    probs = softmax_forward(out_3)
    
    cache = (cache_1, cache_2, cache_3)
    
    return probs, cache
             
def nn_backward(y, probs, cache):
    cache_1, cache_2, cache_3 = cache
    
    dout_3 = softmax_backward(y, probs)
    dout_2, dw2, db2 = affine_backward(dout_3, cache_3)
    dout_1 = relu_backward(dout_2, cache_2)
    dx, dw1, db1 = affine_backward(dout_1, cache_1)
    
    return dx, dw1, db1, dw2, db2

lr = 1e-3
iterations = 10000
for iteration in range(iterations):
    score = 0
    for sample_no in range(N):
        x_i = x[sample_no]
        y_i = y[sample_no]
        
        probs, cache = nn_forward(x_i, w1, b1, w2, b2)
        dx, dw1, db1, dw2, db2 = nn_backward(y_i, probs, cache)
        
        # Update parameters
        w1 += lr * dw1
        b1 += lr * db1
        w2 += lr * dw2
        b2 += lr * db2
        
        # Accuracy
        y_hat_i = np.argmax(probs)
        score += int(y_hat_i == y_i)
    
    print('iteration: {}\tAccuracy: {}'.format(iteration, score / N))