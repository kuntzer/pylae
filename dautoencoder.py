import numpy as np
import copy
import scipy.optimize
from datetime import datetime

import utils as u
import classae
import processing

import logging
logger = logging.getLogger(__name__)

class AutoEncoder(classae.GenericAutoEncoder):
	
	def pre_train(self, data, architecture, layers_activations, mini_batch, iterations, corruption=None, regularisation=0, **kwargs):

		if self.layer_type == "dA" :
			import layers.dA_layer as network
			logger.info("layers.dA_layer used as layer")
		else:
			raise RuntimeError("Layer/pre-training type %s unknown." % self.rbm_type)
		
		if not len(architecture) == len(layers_activations):
			raise ValueError("The size of the list of activation function must match the number of layers.")
		
		layers = []
		shape_previous_layer = np.shape(data)
		for ii in range(len(architecture)):
			logger.info("Pre-training layer {}/{}...".format(ii + 1, len(architecture)))
			if np.shape(corruption) == shape_previous_layer and ii > 0:
				corruption_lvl = layers[ii-1].feedforward(corruption)
			else:
				corruption_lvl = corruption
				
			layer = network.Layer(architecture[ii], layers_activations[ii], corruption_lvl)
			layer.regularisation = regularisation
			layer.train(data, mini_batch=mini_batch, iterations=iterations, **kwargs)
			logger.info("Finished with pre-training layer {} ({}/{})...".format(layer.weights.shape, ii + 1, len(architecture)))
			shape_previous_layer = np.shape(data)
			data = layer.feedforward(data)
	
			layers.append(layer)
		
		self.is_pretrained = True
		self.set_autoencoder(layers)
		logger.info("Finished with pre-training...")
		
	def fine_tune(self, data, iterations, cost_fct, regularisation=0.0, mini_batch=0, 
					method='L-BFGS-B', corruption=None, **kwargs):
		"""
		Pre-training of a dA layer with cross-entropy (hard coded -- at least for now)
		
		:param data: The data to be used
		:param iterations: the number of epochs (i.e. nb of loops of mini_batch)
		:param cost_fct: what cost function to use
		:param mini_batch: number of samples to use per batch
		:param regularisation: the multipler of the L2 regularisation
		:param method: which method of `scipy.optimize.minimize` to use
		
		all remaining `kwargs` are passed to `scipy.optimize.minimize`.
		"""
		
		###########################################################################################
		# Saving some stuff
		self.regularisation = regularisation
		self.mini_batch = mini_batch
		self.Ndata = self.layers[0].Ndata 
		time_start = datetime.now()

		###########################################################################################
		# Optimisation of the weights and biases according to the cost function
		# under the constraint of regularisation and sparsity given by the user 
		options_ = {'maxiter': iterations, 'disp': True}
		# We overwrite these options with any user-specified kwargs:
		options_.update(kwargs)
		
		if self.mini_batch <= 0:
			b_it = 1
		else:
			b_it = self.Ndata / self.mini_batch
			if b_it * self.mini_batch < self.Ndata: b_it += 1

		for i_it in range(b_it):
			
			if b_it > 1:
				time_itstart = datetime.now()
				logger.info("Starting epoch {}/{}...".format(i_it+1, b_it))
			else:
				logger.info("Starting fine-tuning...")

			# Copy the weights and biases into a state vector theta ################################
			weights = []
			biases = []
			for jj in range(self.mid * 2):
				weights.append(copy.copy(self.layers[jj].weights))
				biases.append(self.layers[jj].biases) 
				
			theta, indices, weights_shape, biases_shape = self._roll(weights, biases)
			del weights, biases
			
			self.weights_shape = weights_shape
			self.biases_shape = biases_shape
			self.indices = indices
			########################################################################################
			
			# Selecting the cost function ##########################################################
			if cost_fct == 'cross-entropy':
				J = lambda x: self.xentropy_cost(x, data, corruption)
			else:
				J = lambda x: self.l2_cost(x, data, corruption)
			########################################################################################
			
			# Optimising now #######################################################################
			result = scipy.optimize.minimize(J, theta, method=method, jac=True, options=options_)
						
			if len(result) == 9:
				opt_theta = result.x
				logger.info("Done with optimization, {0} iterations and {1} evaluations of the objective functions".format(result.nit, result.nfev))
			else:
				logger.warning("Optimization output is fishy")	
			########################################################################################
		
			# Unroll the state vector and saves it to self #########################################	
			for jj in range(self.mid * 2):
					
				w, b = self._unroll(opt_theta, jj, indices, weights_shape, biases_shape)
				
				self.layers[jj].weights = w
				self.layers[jj].biases = b
			########################################################################################
			
			if b_it > 1:
				now = datetime.now()
				logger.info("Epoch {}/{} done in {}...".format(i_it+1, b_it, (now - time_itstart)))
		
			now = datetime.now()
			logger.info("Fine-tuning of AE done in {}...".format(now - time_start))
			
		# We're done !
		self.is_trained = True
	
	def xentropy_cost(self, theta, data, corruption):

		# Unrolling the weights and biases
		for jj in range(self.mid * 2):
			w, b = self._unroll(theta, jj, self.indices, self.weights_shape, self.biases_shape)

			self.layers[jj].weights = w
			self.layers[jj].biases = b

		m = data.shape[1]

		# Forward pass
		if corruption is not None:
			cdata = processing.corrupt(self, data, corruption)
			raise NotImplemented("add h = self.feedforward(cdata) and correct algo")
		else:
			cdata = data
		
		h = self.feedforward(data)

		wgrad = []
		bgrad = []
		############################################################################################
		# Compute the gradients:
		# http://neuralnetworksanddeeplearning.com/chap3.html for details
		dEda = None
		
		for jj in range(self.mid * 2 - 1, -1, -1):
			# The output of the layer right before is
			if jj - 1 < 0:
				hn = data
			else:
				hn = self.layers[jj-1].output
			
			# If last layer, we compute the delta = output - expectation
			if dEda is None: 
				dEda = h - data
				dEda = dEda.T
			else:
				if corruption is None:
					a = self.layers[jj].output.T
				else:
					a = self.feedforward_to_layer(cdata, jj).T
				
				wp1 = self.layers[jj+1].weights
				dEda = wp1.dot(dEda) * (a * (1. - a))
				
			dEdb = np.mean(dEda, axis=1)
			dEdw = ((dEda).dot(hn).T + self.regularisation * self.layers[jj].weights) / m
			
			wgrad.append(dEdw)
			bgrad.append(dEdb)

		# Reverse the order since back-propagation goes backwards 
		wgrad = wgrad[::-1]
		bgrad = bgrad[::-1]

		# Computes the cross-entropy
		cost = u.cross_entropy(data, h)

		self.train_history.append(cost)
		
		# Returns the gradient as a vector.
		grad = self._roll(wgrad, bgrad, return_info=False)
		return cost, grad
	
	def l2_cost(self, theta, data, corruption):

		# Unrolling the weights and biases
		for jj in range(self.mid * 2):
			w, b = self._unroll(theta, jj, self.indices, self.weights_shape, self.biases_shape)

			self.layers[jj].weights = w
			self.layers[jj].biases = b
	
		# Number of training examples
		m = data.shape[1]

		# Forward pass
		if corruption is not None:
			cdata = processing.corrupt(self, data, corruption)
			ch = self.feedforward(cdata)
		else:
			cdata = data
		
		h = self.feedforward(data)
		if corruption is None:
			ch = self.feedforward(cdata)

		wgrad = []
		bgrad = []
		
		# Back-propagation
		delta = -(data - ch)
	
		# Compute the gradient:
		for jj in range(self.mid * 2 - 1, -1, -1):
			if jj < self.mid * 2 - 1:

				delta = (self.layers[jj+1].weights.dot(delta.T)).T
				delta = np.array(delta)
			
			delta *= self.layers[jj].activation_fct_prime(self.layers[jj].activation)
			
			#dEdw = ((dEda.T).dot(data).T + self.regularisation * self.weights) / m
			#grad_w = delta.dot(self.layers[jj].input) / m + self.regularisation * self.layers[jj].weights
			grad_w = (((self.layers[jj].input.T).dot(delta)) + self.regularisation * self.layers[jj].weights )/ m
			grad_b = np.mean(delta, axis=0)
			wgrad.append(grad_w)
			bgrad.append(grad_b)
				
		# Reverse the order since back-propagation goes backwards 
		wgrad = wgrad[::-1]
		bgrad = bgrad[::-1]
		
		# Computes the L2 norm + regularisation
		cost = np.sum((h - data) ** 2) / (2 * m) + (self.regularisation / 2) * \
			(sum([((self.layers[jj].weights)**2).sum() for jj in range(self.mid * 2)]))
		
		self.train_history.append(cost)
		
		# Returns the gradient as a vector.
		grad = self._roll(wgrad, bgrad, return_info=False)
		return cost, grad
	