import numpy as np
import copy
import os
import scipy.optimize

import utils as u
import classae
import processing

class AutoEncoder(classae.GenericAutoEncoder):
	
	def pre_train(self, data, architecture, layers_activations, mini_batch, iterations, corruption=None, regularisation=0, **kwargs):

		if self.layer_type == "dA" :
			import layers.dA_layer as network
		elif self.layer_type == "rmse" :
			import layers.rmse_layer as network
		else:
			raise RuntimeError("Layer/pre-training type %s unknown." % self.rbm_type)
		
		if not len(architecture) == len(layers_activations):
			raise ValueError("The size of the list of activation function must match the number of layers.")
		
		layers = []
		shape_previous_layer = np.shape(data)
		for ii in range(len(architecture)):
			print "Pre-training layer %d..." % (ii + 1)
			if np.shape(corruption) == shape_previous_layer and ii > 0:
				corruption_lvl = layers[ii-1].feedforward(corruption)
			else:
				corruption_lvl = corruption
				
			layer = network.Layer(architecture[ii], layers_activations[ii], corruption_lvl, verbose=self.verbose)
			layer.regularisation = regularisation
			layer.train(data, mini_batch=mini_batch, iterations=iterations, **kwargs)
			print 'continue with next level'
			shape_previous_layer = np.shape(data)
			data = layer.feedforward(data)
	
			layers.append(layer)
			#raise RuntimeError("Stopping heer")
			
		
		self.is_pretrained = True
		self.set_autoencoder(layers)
		print 'Pre-training complete.'
		
	def fine_tune(self, data, iterations=400, regularisation=0.0, mini_batch=0, 
					method='L-BFGS-B', verbose=True, corruption=None, cost_fct='L2', **kwargs):
		"""
		Pre-training of a dA layer with cross-entropy (hard coded -- at least for now)
		
		:param data: The data to be used
		:param iterations: the number of epochs (i.e. nb of loops of mini_batch)
		:param mini_batch: number of samples to use per batch
		:param regularisation: the multipler of the L2 regularisation
		:param method: which method of `scipy.optimize.minimize` to use
		
		all remaining `kwargs` are passed to `scipy.optimize.minimize`.
		"""

		# TODO:
		# we could change this in the whole code so that we don't need to transpose too many
		# times. However, transpose is an a fast operation as it returns a view of the array...
		#data = data.T
		
		###########################################################################################
		# Initialisation of the weights and bias
		
		
		###########################################################################################
		# Saving some stuff
		self.regularisation = regularisation
		self.mini_batch = mini_batch
		self.Ndata = self.layers[0].Ndata 

		###########################################################################################
		# Optimisation of the weights and biases according to the cost function
		# under the constraint of regularisation and sparsity given by the user 
		options_ = {'maxiter': iterations, 'disp': verbose}
		# We overwrite these options with any user-specified kwargs:
		options_.update(kwargs)
		
		if self.mini_batch <= 0:
			b_it = 1
		else:
			b_it = self.Ndata / self.mini_batch
			if b_it * self.mini_batch < self.Ndata: b_it += 1

		for i_it in range(b_it):
			if self.verbose: print "** Starting epoch {}/{}... **".format(i_it+1, b_it)
			# Copy the weights and biases into a state vector theta
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
			
			if cost_fct == 'cross-entropy':
				J = lambda x: self.xcross_cost(x, data, corruption)
			else:
				raise NotImplemented()
			
			result = scipy.optimize.minimize(J, theta, method=method, jac=True, options=options_)
			opt_theta = result.x
		
			###########################################################################################
			# Unroll the state vector and saves it to self.	
			for jj in range(self.mid * 2):
					
				w, b = self._unroll(opt_theta, jj, indices, weights_shape, biases_shape)
				
				self.layers[jj].weights = w
				self.layers[jj].biases = b
			
			if verbose: print result
		
		# We're done !
		self.is_trained = True
	
	def xcross_cost(self, theta, data, corruption, log_cost=True):

		# Unrolling the weights and biases
		for jj in range(self.mid * 2):
			w, b = self._unroll(theta, jj, self.indices, self.weights_shape, self.biases_shape)

			self.layers[jj].weights = w
			self.layers[jj].biases = b

		batch = data
		m = batch.shape[1]

		# Forward pass
		if corruption is not None:
			cdata = processing.corrupt(self, batch, corruption)
		else:
			cdata = batch
		
		h = self.feedforward(batch)

		wgrad = []
		bgrad = []
		############################################################################################
		# Compute the gradients:
		# http://neuralnetworksanddeeplearning.com/chap3.html for details
		dEda = None
		
		for jj in range(self.mid * 2 - 1, -1, -1):
			# The output of the layer right before is
			if jj - 1 < 0:
				hn = batch
			else:
				hn = self.layers[jj-1].output
			
			# If last layer, we compute the delta = output - expectation
			if dEda is None: 
				dEda = h - batch
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
		cost = u.cross_entropy(batch, h)

		if log_cost:
			self.train_history.append(cost)
		
		# Returns the gradient as a vector.
		grad = self._roll(wgrad, bgrad, return_info=False)
		return cost, grad
	
	def cost(self, theta, indices, weights_shape, biases_shape, lambda_, sparsity, beta,\
			 data, corruption, cost_fct, dropout, log_cost=True):

		raise ValueError("Don't use this!")
		if cost_fct == 'cross-entropy':
			if beta != 0 or sparsity != 0:
				beta = 0
				sparsity = 0
				#print 'WARNING: Cross-entropy does not support sparsity'

		# Unrolling the weights and biases
		for jj in range(self.mid * 2):
			w, b = self._unroll(theta, jj, indices, weights_shape, biases_shape)

			self.layers[jj].weights = w
			self.layers[jj].biases = b
	
		# Number of training examples
		m = data.shape[1]

		# Forward pass
			
		if corruption is not None:
			cdata = processing.corrupt(self, data, corruption)
		else:
			cdata = data
		ch = self.feedforward(cdata.T, dropout=dropout).T
		h = self.feedforward(data.T, dropout=dropout).T

		# Sparsity
		sparsity_cost = 0
		
		wgrad = []
		bgrad = []
		
		############################################################################################
		# Cost function

		if cost_fct == 'L2':
			raise ValueError("To redo")
			# Back-propagation
			delta = -(data - ch)
		
			# Compute the gradient:
			for jj in range(self.mid * 2 - 1, -1, -1):
				if jj < self.mid * 2 - 1:

					hn = self.layers[jj].output.T.shape[0]
					rho_hat = np.mean(self.layers[jj].output.T, axis=1)

					if beta == 0:
						sparsity_grad = 0
						sparsity_cost = 0
					else:
						rho = sparsity
						
						sparsity_cost += beta * np.sum(u.KL_divergence(rho, rho_hat))
						sparsity_grad = beta * u.KL_prime(rho, rho_hat)
						sparsity_grad = np.matrix(sparsity_grad).T
						#spars_grad = np.tile(spars_grad, m).reshape(m,self.hidden_nodes).T
						#print rho_hat.mean(), 'cost:', sparsity_cost, '<<<<<<<<<<<<<<<'
						
						sparsity_cost += beta * np.sum(u.KL_divergence(rho, rho_hat))
	
					delta = self.layers[jj+1].weights.dot(delta) + beta * sparsity_grad
					delta = np.array(delta)
				
				if self.layers[jj].hidden_type == 'SIGMOID':
					delta *= u.sigmoid_prime(self.layers[jj].activation.T)
				elif self.layers[jj].hidden_type == 'RELU':
					delta *= u.relu_prime(self.layers[jj].activation.T)
				elif self.layers[jj].hidden_type == 'LEAKY_RELU':
					delta *= u.leaky_relu_prime(self.layers[jj].activation.T)
				elif self.layers[jj].hidden_type == 'LINEAR':
					pass 
				else:
					raise ValueError("Unknown activation function %s" % self.layers[jj].hidden_type)
				
				grad_w = delta.dot(self.layers[jj].input) / m + lambda_ * self.layers[jj].weights.T
				grad_b = np.mean(delta, axis=1)
				wgrad.append(grad_w.T)
				bgrad.append(grad_b)
					
			# Reverse the order since back-propagation goes backwards 
			wgrad = wgrad[::-1]
			bgrad = bgrad[::-1]
			
			# Computes the L2 norm + regularisation
			#TODO: COST MISSES THE COMPLETE SPARSITY !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
			cost = np.sum((h - data) ** 2) / (2 * m) + (lambda_ / 2) * \
				(sum([((self.layers[jj].weights)**2).sum() for jj in range(self.mid * 2)])) + \
				sparsity_cost
			#print 'tot cost', cost
		elif cost_fct == 'cross-entropy':
			# Compute the gradients:
			# http://neuralnetworksanddeeplearning.com/chap3.html for details
			dEda = None
			
			for jj in range(self.mid * 2 - 1, -1, -1):
				#print jj, '-------' * 6
				# The output of the layer right before is
				if jj - 1 < 0:
					hn = data.T
				else:
					hn = self.layers[jj-1].output
				
				# If last layer, we compute the delta = output - expectation
				if dEda is None: 
					dEda = ch - data
				else:
					wp1 = self.layers[jj+1].weights
					if corruption is None:
						a = self.layers[jj].output
					else:
						a = self.feedforward_to_layer(cdata.T, jj)
					dEda = wp1.dot(dEda) * (a * (1. - a)).T
					
				dEdb = np.mean(dEda, axis=1)

				dEdw = (dEda.dot(hn) + lambda_ * self.layers[jj].weights.T) / m
				dEdw = dEdw.T
				
				wgrad.append(dEdw)
				bgrad.append(dEdb)
	
			# Reverse the order since back-propagation goes backwards 
			wgrad = wgrad[::-1]
			bgrad = bgrad[::-1]

			# Computes the cross-entropy
			cost = - np.sum(data * np.log(ch) + (1. - data) * np.log(1. - ch), axis=0) 
			cost = np.mean(cost)
			
		if log_cost:
			self.train_history.append(cost)
		
		#exit()
		# Returns the gradient as a vector.
		grad = self._roll(wgrad, bgrad, return_info=False)
		return cost, grad
	
