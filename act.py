"""
Activation functions
"""

import numpy as np

import logging
logger = logging.getLogger(__name__)

def sigmoid(x):
	return 1.0 / (1.0 + np.exp(-x))

def sigmoid_prime(x):
	return sigmoid(x) * (1. - sigmoid(x))

def sigmoideven(x):
	return 2.0 * sigmoid(x) - 1

def sigmoideven_prime(x):
	return 2.0 * sigmoid_prime.prime(x)

def tanh(x):
	return np.tanh(x)

def tanh_prime(x):
	return 1. - tanh(x) * tanh(x)
	
def linear(x):
	return x

def linear_prime(x):
	return np.ones_like(x)
	
def relu(x):
	return np.maximum(np.zeros_like(x), x)
	
def relu_prime(x):
	res = np.ones_like(x) * x
	res[x <= 0] = 0.
	res[x > 0] = 1.
	
	return res

if __name__ == "__main__":
	
	import matplotlib.pyplot as plt
	
	x = np.linspace(-5, 5, 1000)
	acts = [sigmoid, sigmoideven, tanh, identity, relu]
	
	for act in acts:
		plt.plot(x, act(x), label=act.__name__)
		
	plt.xlabel(r"$x$")
	plt.ylabel(r"$f(x)$")
	plt.title("Activation functions")
	plt.ylim(-1.2, 1.2)
	plt.legend()
	plt.grid()
	plt.show()
	