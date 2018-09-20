#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 6 15:21:19 2018

@author: aananya

Linear Regression
"""

import numpy as np
import matplotlib.pyplot as plt 

X = 2 * np.random.rand(10, 1)
Y = 4 + 3 * X + np.random.randn(10, 1)
plt.plot(X,Y,"b.")
plt.show()
X_b=np.c_[np.ones((10,1)), X]
theta_best=np.linalg.inv(X_b.T.dot(X_b)).dot(X_b.T).dot(Y)
print (theta_best)
X_new = np.array([[0], [2]])
X_new_b = np.c_[np.ones((2, 1)), X_new] # add x0 = 1 to each instance
Y_predict = X_new_b.dot(theta_best)
print (Y_predict)
plt.plot(X_new, Y_predict, "r-")
plt.plot(X, Y, "b.")
plt.axis([0, 2, 0, 15])
plt.show()
