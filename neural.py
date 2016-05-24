# -*- coding: utf-8 -*-
"""
Created on Mon May 9 01:32:37 2016

@author: Eng_Ramez
"""

import numpy as np

# sigmoid function and its derivative    
def segmoid(x):
    return 1/(1+np.exp(-x))

def deriv(x):
        return x*(1-x)
    
X = np.array([  [0,0,1],
                [0,1,1],
                [1,0,0],
                [1,1,0] ])
    
y = np.array([[0,1,1,1]]).T


syn0 = np.random.random((3,1))
# max epoch is set to be 1000
for iter in range(1000):

    # forward propagation
    l0 = X
    l1 = segmoid(np.dot(l0,syn0))

    # miss?
    l1_error = y - l1
 
    # slope of the sigmoid at the values in l1
    l1_delta = l1_error * deriv(l1)

    # update weights
    syn0 += np.dot(l0.T,l1_delta)

print ("Output After Training:")
print ((l1))

print("Testing Result")
inp=np.array([1,0,1])
test= segmoid(np.dot(inp,syn0))
print((test))