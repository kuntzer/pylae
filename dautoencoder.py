import numpy as np
import copy
import os
import scipy.optimize

import utils as u
import classae
import processing

class AutoEncoder(classae.GenericAutoEncoder):
	
	def pre_train(self, data, architecture, layers_type, mini_batch, iterations, corruption=None, **kwargs):

		if self.layer_type == "dA" :
			import dA_layer as network
		elif self.layer_type == "rmse" :
			import rmse_layer as network
		else:
			raise RuntimeError("Layer/pre-training type %s unknown." % self.rbm_type)
		
		
		assert len(architecture) + 1 == len(layers_type)
		
		layers = []
		shape_previous_layer = np.shape(data)
		for ii in range(len(architecture)):
			print "Pre-training layer %d..." % (ii + 1)
			if np.shape(corruption) == shape_previous_layer and ii > 0:
				corruption_lvl = layers[ii-1].feedforward(corruption)
			else:
				corruption_lvl = corruption
				
			layer = network.Layer(architecture[ii], layers_type[ii], layers_type[ii+1], mini_batch, 
						iterations, corruption_lvl)
			layer.train(data, **kwargs)
			print 'continue with next level'
			shape_previous_layer = np.shape(data)
			data = layer.feedforward(data)
	
			layers.append(layer)
			
		
		self.is_pretrained = True
		self.set_autoencoder(layers)
		print 'Pre-training complete.'
		
	def fine_tune(self, data, iterations=400, regularisation = 0.0, sparsity=0., beta=0., dropout=None,
					method='L-BFGS-B', verbose=True, return_info=False, corruption=None, cost_fct='L2'):

		# TODO:
		# we could change this in the whole code so that we don't need to transpose too many
		# times. However, transpose is an a fast operation as it returns a view of the array...
		data = data.T
		###########################################################################################
		# Initialisation of the weights and bias
		
		# Copy the weights and biases into a state vector theta
		weights = []
		biases = []
		for jj in range(self.mid * 2):
			weights.append(copy.copy(self.layers[jj].weights))
			biases.append(self.layers[jj].hidden_biases) 
			
		theta, indices, weights_shape, biases_shape = self._roll(weights, biases)
		del weights, biases
		

		###########################################################################################
		# Optimisation of the weights and biases according to the cost function
		# under the constraint of regularisation and sparsity given by the user 
		J = lambda x: self.cost(x, indices, weights_shape, biases_shape,
			regularisation, sparsity, beta, data, corruption, cost_fct=cost_fct, dropout=dropout)
		
		if iterations >= 0:
			options_ = {'maxiter': iterations, 'disp': verbose, 'ftol' : 10. * np.finfo(float).eps, 'gtol': 1e-9}
			result = scipy.optimize.minimize(J, theta, method=method, jac=True, options=options_)
			opt_theta = result.x
		else:
			c, _ = J(theta)
			return c
		
		###########################################################################################
		# Unroll the state vector and saves it to self.	
		for jj in range(self.mid * 2):
				
			w, b = self._unroll(opt_theta, jj, indices, weights_shape, biases_shape)
			
			self.layers[jj].weights = w
			self.layers[jj].hidden_biases = b
		
		if verbose: print result
		
		# We're done !
		self.is_trained = True
		
		# If the user requests the informations about the minimisation...
		if return_info: return result
	
	def cost(self, theta, indices, weights_shape, biases_shape, lambda_, sparsity, beta,\
			 data, corruption, cost_fct, dropout, log_cost=True):

		if cost_fct == 'cross-entropy':
			if beta != 0 or sparsity != 0:
				beta = 0
				sparsity = 0
				#print 'WARNING: Cross-entropy does not support sparsity'

		# Unrolling the weights and biases
		for jj in range(self.mid * 2):
			w, b = self._unroll(theta, jj, indices, weights_shape, biases_shape)

			self.layers[jj].weights = w
			self.layers[jj].hidden_biases = b
	
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

				dEdw = dEda.dot(hn) / m + lambda_ * self.layers[jj].weights.T
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
	
	def get_cost(self, data, regularisation = 0., sparsity=0., beta=0.,
					method='L-BFGS-B', verbose=True, return_info=False, cost_fct='L2'):
		"""
		Returns the cost, but this is very slow!!!
		"""
		
		iterations = -1
		return self.fine_tune(data, iterations, regularisation, sparsity, beta,	method, verbose, return_info, cost_fct)

	def sgd(self, data, iterations, learning_rate, initial_momentum, final_momentum, minibatch=10, \
		annealing=None, max_epoch_without_improvement=30, early_stop=True, corruption=None):
		"""
		Performes an Stochastic gradient descent (SGD) optimisation of the network.
		"""
		
		m = data.shape[1]
		data = data.T
		
		###########################################################################################
		# Initialisation of the weights and bias
		
		# Copy the weights and biases into a state vector theta
		weights = []
		biases = []
		for jj in range(self.mid * 2):
			weights.append(copy.copy(self.layers[jj].weights))
			biases.append(self.layers[jj].hidden_biases) 
			
		theta, indices, weights_shape, biases_shape = self._roll(weights, biases)
		del weights, biases
		
		###########################################################################################
		v_mom = 0
		best_cost = 1e8
		
		batch_indices = np.arange(m)
		n_minibatches = np.int(np.ceil(m/minibatch))
		
		gamma = initial_momentum
		
		for epoch in range(iterations):
			np.random.shuffle(batch_indices)			
			for ibatch in range(n_minibatches+1):
				ids = batch_indices[ibatch*minibatch:(ibatch+1)*minibatch]
				batch = data[:,ids]
				
				_, thetan = self.cost(theta, indices, weights_shape, biases_shape,
				0, 0, 0, batch, corruption=corruption, cost_fct='cross-entropy', log_cost=False)
				v_mom = gamma * v_mom + learning_rate * thetan
				theta -= v_mom
			
			actual = self.feedforward(data.T)
			cost = u.cross_entropy(data.T, actual)
			print 'Epoch %4d/%4d:\t%e' % (epoch+1, iterations, cost)

			self.train_history.append(cost)
			
			if cost <= best_cost :
				best_cost = cost
				iter_best = epoch
				
			if epoch - iter_best > max_epoch_without_improvement :
				print 'STOP: %d epoches without improvment' % max_epoch_without_improvement
				break
			
			
			if annealing is not None:
				learning_rate /= (1. + float(epoch) / annealing)
				
			if epoch > 100:
				gamma = final_momentum
			else:
				gamma = initial_momentum + (final_momentum - initial_momentum) * u.sigmoid(epoch - 50)
			print learning_rate, gamma
			
		###########################################################################################
		# Unroll the state vector and saves it to self.	
		for jj in range(self.mid * 2):
			w, b = self._unroll(theta, jj, indices, weights_shape, biases_shape)
			
			self.layers[jj].weights = w
			self.layers[jj].hidden_biases = b
			
		# We're done !
		self.is_trained = True
