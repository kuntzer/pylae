import numpy as np
import utils
import copy
import os
import logging

logger = logging.getLogger(__name__)

class GenericAutoEncoder():
	
	def __init__(self, name='ae', layer_type="dA", directory=''):
		self.name = name
		self.is_pretrained = False
		self.is_trained = False
		self.layer_type = layer_type
		
		self.filepath = os.path.join(directory, name, layer_type)
		if not os.path.isdir(self.filepath):
			os.makedirs(self.filepath)
		self.directory = directory
		self.train_history = []
		
	def info(self):
		"""
		Prints out all the variables for a given galaxy of the class
		"""
		import inspect
	
		message = "All variables available for this autoencoder"		
		print message
		print '-'*len(message)
		attributes = inspect.getmembers(self, lambda a:not(inspect.isroutine(a)))
		for a in attributes:
			if (a[0].startswith('__') and a[0].endswith('__')): continue
			print a[0], "=", a[1]
			
	def _clear_memory(self):
		self.rbms = None
			
		for jj in range(len(self.layers)):
			self.layers[jj].output = None
			self.layers[jj].activation = None
			self.layers[jj].input = None
		
	def set_autoencoder(self, encoder):
		"""
		From a list of layers object `encoder`, create an autoencoder by adding in a reversed order 
		of the encoder list to the encoder. Swaps the weights and biases too.
		:param encoder: A list of pre-trained RBMs in the encoder order.
		:type list:
		"""
		self.rbms = encoder
		
		encoder = copy.deepcopy(encoder)
		decoder = copy.deepcopy(encoder)
		decoder = decoder[::-1]
		self.layers = encoder + decoder
		
		# Swap the values for the decoder part
		self.mid = len(self.layers)/2
		for layer in self.layers[self.mid:] :
			layer.weights = layer.weights.T
			layer.biases, layer.inverse_biases = layer.inverse_biases, layer.biases		
			
	def feedforward(self, data):
		"""
		Encodes and decodes the data using the full auto-encoder
		:param data: The data in the same format than the training data
		
		:returns: the fed-forward data
		"""
		for layer in self.layers :
			data = layer.feedforward_memory(data)

		return data
	
	def feedforward_to_layer(self, data, j):
		"""
		Encodes and decodes the data using the full autoencoder
		:param data: The data in the same format than the training data
		:param j: the layer to stop at 
		
		:returns: the fed-forward data
		"""
		for layer in self.layers[:j+1] :
			data = layer.feedforward(data)

		return data
		
	def encode(self, data):
		"""
		Encodes the data using the encoder
		:param data: The data in the same format than the training data
		
		:returns: encoded data
		"""
		for layer in self.layers[:self.mid] :
			logger.info("encoding, {}".format(layer.weights.shape))
			data = layer.feedforward(data)
			
		return data
		
	def decode(self, data):
		"""
		Decodes the data using the decoder
		:param data: encoded data
		
		:returns: decoded data
		"""
		for layer in self.layers[self.mid:] :
			logger.info("decoding, {}".format(layer.weights.shape))
			data = layer.feedforward(data)
			
		return data

	def _roll(self, weightslist, biaseslist, return_info=True):
		biases_shape = []
		biases = []
		weights_shape = []
		weights = []
		indices = np.array([0])
		
		# Copy the weights and biases into a state vector theta
		for jj in range(self.mid * 2):
			w = weightslist[jj].T
			b = biaseslist[jj]
			biases = np.concatenate((biases, b))
			biases_shape.append(np.shape(b)[0])
			
			weights_shape.append(np.shape(w))
			reshaped = weights_shape[-1][0] * weights_shape[-1][1]
			w = w.reshape(reshaped)

			indices = np.concatenate((indices, [indices[-1]+reshaped]))
			weights = np.concatenate((weights, w))
			
		theta = np.concatenate((weights, biases))
		
		# Let's remember the indices for simplicity
		for ib in biases_shape:
			indices = np.concatenate((indices, [indices[-1]+ib]))
		
		if return_info:
			return theta, indices, weights_shape, biases_shape
		else:
			return theta
		
	def _unroll(self, theta, layer, indices, weights_shape, biases_shape):
		w = theta[indices[layer]: indices[layer+1]].reshape(weights_shape[layer][0], weights_shape[layer][1]).transpose() 
		b = theta[indices[layer + 2 * self.mid]: indices[layer + 1 + 2 * self.mid]].reshape(biases_shape[layer])

		return w, b
	
	def save(self, filepath=None, clear_memory=True):
		if filepath is None:
			filepath = self.filepath
			
		if clear_memory:
			self._clear_memory()
			
		utils.writepickle(self, os.path.join(filepath, 'ae.pkl'))
