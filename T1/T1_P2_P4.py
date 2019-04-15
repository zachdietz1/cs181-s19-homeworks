#####################
# CS 181, Spring 2019
# Homework 1, Problem 2, Part 4
#
##################

import numpy as np

data = np.matrix('0 , 0 , 0; 0 , .5 , 0; 0 , 1 , 0; .5 , 0 , .5; .5 , .5 , .5; .5 , 1 , .5; 1 , 0 , 1; 1 , .5 , 1; 1 , 1 , 1')
alpha = 10
k1 = np.multiply(alpha, np.matrix('1, 0; 0, 1'))
k2 = np.multiply(alpha, np.matrix('.1, 0; 0, 1'))
k3 = np.multiply(alpha, np.matrix('1, 0; 0, .1'))

def K(x, x_, W):
	dif = np.subtract(x, x_)
	partial = np.dot(W,dif)
	return np.exp(- np.dot(np.transpose(dif), partial))
	
def f(x, W, index):
	top, bottom = 0, 0
	for i, row in enumerate(data):
		if i != index:
			xn_, yn = np.transpose(row)[0:2], np.transpose(row)[2]
			top += K(xn_,x,W) * yn
			bottom += K(xn_,x,W)
	return top / bottom

def loss(W):
	squared_loss = 0
	for i, row in enumerate(data):
		xn_, yn = np.transpose(row)[0:2], np.transpose(row)[2]
		squared_loss += (f(xn_, W, i) - yn) ** 2
	return squared_loss

print("loss for W_1: {}".format(loss(k1)))
print("loss for W_2: {}".format(loss(k2)))
print("loss for W_3: {}".format(loss(k3)))