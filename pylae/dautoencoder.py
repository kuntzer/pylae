import numpy as np
import utils
import copy
import os
import classae
import scipy.optimize

class AutoEncoder(classae.GenericAutoEncoder):
	
	def __init__(self, name='ae', layer_type="dA", directory='', verbose=False):
		self.name = name
		self.is_pretrained = False
		self.is_trained = False
		self.verbose = verbose
		self.layer_type = layer_type
		
		self.filepath = os.path.join(directory, name, layer_type)
		if not os.path.isdir(self.filepath):
			os.makedirs(self.filepath)
		self.directory = directory
		self.train_history = []
	
	def pre_train(self, data, architecture, layers_type, mini_batch, iterations, corruption=None):

		if self.layer_type == "dA" :
			import dA_layer as network
		else:
			raise RuntimeError("Layer/pre-training type %s unknown." % self.rbm_type)
		
		
		assert len(architecture) + 1 == len(layers_type)
		
		layers = []
		for ii in range(len(architecture)):
			print "Pre-training layer %d..." % (ii + 1)
			layer = network.Layer(architecture[ii], layers_type[ii], layers_type[ii+1], mini_batch, 
						iterations, corruption)
			layer.train(data)
			print 'continue with next level'
			data = layer.feedforward(data)
	
			layers.append(layer)
			
		
		self.is_pretrained = True
		self.set_autoencoder(layers)
	
	def fine_tune(self, data, iterations=400, regularisation = 0.003, sparsity=0.1, beta=3.,
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
			regularisation, sparsity, beta, data, corruption, cost_fct=cost_fct)
		
		if iterations >= 0:
			options_ = {'maxiter': iterations, 'disp': verbose}
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
	
	def _corrupt(self, data, corruption):
		
		scales = np.random.uniform(low=corruption[0], high=corruption[1], size=data.shape[1])
		
		noise_maps = [np.random.normal(scale=sig, size=data.shape[0]) for sig in scales]
		noise_maps = np.asarray(noise_maps)
		
		cdata = data + noise_maps.T
		
		
		print np.amin(cdata), np.amax(cdata)
		return cdata
	
	def cost(self, theta, indices, weights_shape, biases_shape, lambda_, sparsity, beta,\
			 data, corruption, cost_fct, log_cost=True):

		if cost_fct == 'cross-entropy':
			if beta != 0 or sparsity != 0:
				beta = 0
				sparsity = 0
				print 'WARNING: Cross-entropy does not support sparsity'

		# Unrolling the weights and biases
		for jj in range(self.mid * 2):
			w, b = self._unroll(theta, jj, indices, weights_shape, biases_shape)

			self.layers[jj].weights = w
			self.layers[jj].hidden_biases = b
	
		# Number of training examples
		m = data.shape[1]

		# Forward pass
			
		if corruption is not None:
			cdata = self._corrupt(data, corruption)
		else:
			cdata = data
		ch = self.feedforward(cdata.T).T
		h = self.feedforward(data.T).T

		# Sparsity
		sparsity_cost = 0
		
		wgrad = []
		bgrad = []
		
		############################################################################################
		# Cost function

		if cost_fct == 'L2':
			
			if corruption is not None:
				raise NotImplemented('corruption not covered in L2')
			
			# Back-propagation
			delta = -(data - h)
		
			# Compute the gradient:
			for jj in range(self.mid * 2 - 1, -1, -1):

				if jj < self.mid * 2 - 1:
					# TODO: Sparsity: do we want it at every (hidden) layer ?? 
					"""print jj
					print np.shape(self.layers[2].output.T)
					
					print hn
					print self.layers[2].hidden_nodes
					print m 
					exit()"""
					hn = self.layers[jj].output.T.shape[0]
					#print hn, np.shape(self.layers[2].output.T)
					rho_hat = np.mean(self.layers[jj].output.T, axis=1)
					rho = np.tile(sparsity, hn)
					#print np.shape(rho_hat), rho.shape
					sparsity_delta = np.tile(- rho / rho_hat + (1 - rho) / (1 - rho_hat), (m, 1)).transpose()
					sparsity_cost += beta * np.sum(utils.KL_divergence(rho, rho_hat))
	
					delta = self.layers[jj+1].weights.dot(delta) + beta * sparsity_delta
					
				delta *= utils.sigmoid_prime(self.layers[jj].activation.T)
				
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

				dEdw = dEda.dot(hn) / m
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
