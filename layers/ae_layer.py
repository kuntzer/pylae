import numpy as np
import scipy.optimize
from datetime import datetime

import layer
from .. import processing
from .. import act 
from .. import utils
import logging

logger = logging.getLogger(__name__)

class Layer(layer.AE_layer):
	def __init__(self, hidden_nodes, activation, debug):
		"""
		:param hidden_nodes: Number of neurons in layer
		:param activation: Activation function for the layer
		:param mini_batch: number of training sample
		:param max_epoch_without_improvement: how many iterations should be done after best performance is 
			reached to probe the rmsd behaviour. 
		:param early_stop: if True, stops the iterations when there is no improvements anymore, 
			makes probes the `max_epoch_without_improvement` following iterations before stopping.
		:param debug: make a slow but useful gradient checking operations.
		"""
		self.hidden_nodes = hidden_nodes
		self.activation_name = activation
		self.train_history = []
		self.debug = debug
		self.it = 0
		
		self.activation_fct = eval("act.{}".format(self.activation_name.lower()))
		self.activation_fct_prime = eval("act.{}_prime".format(self.activation_name.lower()))
		
	def xentropy_cost(self, theta, data):
		"""
		Computes the L2 cost and gradient
		http://neuralnetworksanddeeplearning.com/chap3.html for details
		"""
		
		m, _ = data.shape

		# Unroll theta
		self.weights, self.inverse_biases, self.biases = self._unroll(theta)

		# Forward passes
		
		ch = self.round_feedforward(data)
		hn = self.output
			
		# Compute the gradients:
		# http://neuralnetworksanddeeplearning.com/chap3.html for details
		
		# First: bottom to top
		dEda = ch - data
		dEdvb = np.mean(dEda, axis=0)

		# Second: top to bottom
		dEda = (dEda).dot(self.weights) * (hn * (1. - hn))
		dEdhb = np.mean(dEda, axis=0)
		
		dEdw = ((dEda.T).dot(data).T) / m + self.regularisation * self.weights
		grad = self._roll(dEdw, dEdvb, dEdhb)

		# Computes the cross-entropy
		self.current_cost_value = utils.cross_entropy(data, ch)
		

		if self.debug and self.it > 2:
			epsilon = 1e-4
			
			print "-id-|--grad---|num-grad-|-delta-|-mult--|status"
			print "----|---------|---------|-------|-------|------"
			
			debug_res = True
			for i in range(np.size(theta)):
				basis_vectors = np.zeros_like(theta)
				basis_vectors[i] = 1
				tetha_p = theta + epsilon * basis_vectors
				tetha_m = theta - epsilon * basis_vectors
				
				self.weights, self.inverse_biases, self.biases = self._unroll(tetha_p)
				h = self.round_feedforward(data)
				cost_p = utils.cross_entropy(data, h) 
				
				self.weights, self.inverse_biases, self.biases = self._unroll(tetha_m)
				h = self.round_feedforward(data)
				cost_m = utils.cross_entropy(data, h) 
				
				num_grad = 0.5 * (cost_p - cost_m) / epsilon
				
				status = np.abs(num_grad - grad[i]) < epsilon
				
				if not status:
					debug_res = False
				print "{0:04d}|{1:+0.6f}|{2:+0.6f}|{3:0.5f}|{4:0.5f}|{5}".format(i, grad[i], num_grad, np.abs(num_grad - grad[i]), num_grad / grad[i], status)
			
			print 'Final status of debug:', debug_res
		
		# Returns the gradient as a vector.
		return self.current_cost_value, grad
	
	def l2_cost(self, theta, data):

		m, _ = data.shape

		# Unroll theta
		self.weights, self.inverse_biases, self.biases = self._unroll(theta)
		# Forward passes
		
		h, activation_last = self.round_feedforward(data, return_activation=True)
			
		# Back-propagation
		prime = self.activation_fct_prime(activation_last)
		deltal = (h - data) * prime
		dEdvb = np.mean(deltal, axis=0)
		
		prime = self.activation_fct_prime(self.activation)
		deltal = np.dot(deltal, self.weights) * prime
		dEdhb = np.mean(deltal, axis=0)
		
		dEdw = (h.T.dot(deltal)) / m + self.regularisation * self.weights
		grad = self._roll(dEdw, dEdvb, dEdhb)
		
		self.current_cost_value = np.sum((h - data) ** 2) / (2 * m) + ((self.regularisation / 2) * np.abs(self.weights)**2).sum()

		if self.debug and self.it > 2:
			epsilon = 1e-4
			
			print "-id-|--grad---|num-grad-|-delta-|-ratio-|status"
			print "----|---------|---------|-------|-------|------"
			
			debug_res = True
			for i in range(np.size(theta)):
				basis_vectors = np.zeros_like(theta)
				basis_vectors[i] = 1
				tetha_p = theta + epsilon * basis_vectors
				tetha_m = theta - epsilon * basis_vectors
				
				self.weights, self.inverse_biases, self.biases = self._unroll(tetha_p)
				h = self.round_feedforward(data)
				cost_p = np.sum((h - data) ** 2) / (2 * m)
				
				self.weights, self.inverse_biases, self.biases = self._unroll(tetha_m)
				h = self.round_feedforward(data)
				cost_m = np.sum((h - data) ** 2) / (2 * m)
				
				num_grad = 0.5 * (cost_p - cost_m) / epsilon
				
				status = np.abs(num_grad - grad[i]) < epsilon
				
				if not status:
					debug_res = False
				print "{0:04d}|{1:+0.6f}|{2:+0.6f}|{3:0.5f}|{4:0.5f}|{5}".format(i, grad[i], num_grad, np.abs(num_grad - grad[i]), num_grad / grad[i], status)
			
			print 'Final status of debug:', debug_res
		
		# Returns the gradient as a vector.
		return self.current_cost_value, grad
	
	def _log(self, *args, **kwargs):
		"""
		Used as a callback function for the minization algorithm
		"""
		self.train_history.append(self.current_cost_value)
		self.it += 1
		
	def gd(self, data, iterations, cost_fct, learning_rate=0.1, regularisation=0, **kwargs):
		"""
		Pre-training of a dA layer with cross-entropy (hard coded -- at least for now)
		
		:param data: The data to be used
		:param iterations: the number of iterations
		:param cost_fct: what cost function to use
		:param mini_batch: number of samples to use per batch
		:param regularisation: the multipler of the L2 regularisation
		:param method: which method of `scipy.optimize.minimize` to use
		:pram weights: the standard deviation of the zero-centred weights (biases are initialised to 0)
		
		all remaining `kwargs` are passed to `scipy.optimize.minimize`.
		"""
		
		self.Ndata, numdims = np.shape(data)
		
		time_start = datetime.now()

		for it in range(self.iterations):
			theta = self._roll(self.weights, self.inverse_biases, self.biases)
			# Selecting the cost function ##########################################################
			if cost_fct == 'cross-entropy':
				cost, grad = self.xentropy_cost(theta, data)
			elif cost_fct == 'L2':
				cost, grad = self.l2_cost(theta, data)
			else:
				raise ValueError("Cost function {} unknown".format(cost_fct))
			########################################################################################
			logger.info("{}it: {:0.6f}".format(it, cost))
			dweights, dinverse_biases, dbiases = self._unroll(grad)
			self.weights -= dweights * learning_rate
			self.inverse_biases -= dinverse_biases * learning_rate
			self.biases -= dbiases * learning_rate
			theta = self._roll(self.weights, self.inverse_biases, self.biases)
			self._log()
		now = datetime.now()
		logger.info("Pre-training of layer {} done in {}...".format(self.weights.shape, (now - time_start)))
		
	
	def train(self, data, iterations,  cost_fct, mini_batch=0, regularisation=0, method='L-BFGS-B', weight=0.1, **kwargs):
		"""
		Pre-training of a dA layer with cross-entropy (hard coded -- at least for now)
		
		:param data: The data to be used
		:param iterations: the number of iterations
		:param cost_fct: what cost function to use
		:param mini_batch: number of samples to use per batch
		:param regularisation: the multipler of the L2 regularisation
		:param method: which method of `scipy.optimize.minimize` to use
		:pram weights: the standard deviation of the zero-centred weights (biases are initialised to 0)
		
		all remaining `kwargs` are passed to `scipy.optimize.minimize`.
		"""
		
		self.Ndata, numdims = np.shape(data)
		self.mini_batch = mini_batch
		
		self.iterations = iterations
		
		self.visible_dims = numdims
		
		self.weights = weight * np.random.randn(numdims, self.hidden_nodes)
		
		# This apparently this should be better, but definitely noisier (or plain worse)!
		
		self.weights = 4 * np.random.uniform(
					low=-np.sqrt(6. / (numdims + self.hidden_nodes)),
					high=np.sqrt(6. / (numdims + self.hidden_nodes)),
					size=(numdims, self.hidden_nodes))
		"""
		self.weights3 = 2. * np.random.uniform(
					low=-np.sqrt(1. / (numdims)),
					high=np.sqrt(1. / (numdims)),
					size=(numdims, self.hidden_nodes))
		
		#print 'WEIGHTS:', np.amin(self.weights), np.amax(self.weights), np.mean(self.weights), np.std(self.weights)
		"""
		
		self.inverse_biases = np.zeros(numdims)
		self.biases = np.zeros(self.hidden_nodes)
		
		time_start = datetime.now()
		
		if method == "gd":
			# This is a bit special. It is mostly for debug purposes.
			self.gd(data, iterations, cost_fct, regularisation=regularisation, **kwargs)
			return
	
		###########################################################################################
		options_ = {'maxiter': self.iterations, 'disp': True}
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
				logger.info("Starting pre-training...")

			theta = self._roll(self.weights, self.inverse_biases, self.biases)
			
			if self.mini_batch <= 0:
				batch = data
			else:
				ids_batch = utils.select_mini_batch(self)
				batch = data[ids_batch]
				
			# Selecting the cost function ##########################################################
			if cost_fct == 'cross-entropy':
				J = lambda x: self.xentropy_cost(x, batch)
			elif cost_fct == 'L2':
				J = lambda x: self.l2_cost(x, batch)
			else:
				raise ValueError("Cost function {} unknown".format(cost_fct))
			########################################################################################
					
			result = scipy.optimize.minimize(J, theta, method=method, jac=True, options=options_, callback=self._log)
			
			if len(result) == 9:
				opt_theta = result.x
				logger.info("Done with optimization, {0} iterations and {1} evaluations of the objective functions".format(result.nit, result.nfev))
			else:
				logger.warning("Optimization output is fishy")		
			
			###########################################################################################
			# Unroll the state vector and saves it to self.	
			self.weights, self.inverse_biases, self.biases = self._unroll(opt_theta)
		
			if b_it > 1:
				now = datetime.now()
				logger.info("Epoch {}/{} done in {}...".format(i_it+1, b_it, (now - time_itstart)))
		
			now = datetime.now()
			logger.info("Pre-training of layer {} done in {}...".format(self.weights.shape, (now - time_start)))

