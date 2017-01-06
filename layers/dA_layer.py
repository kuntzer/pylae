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
	def __init__(self, hidden_nodes, activation, corruption):
		"""
		:param hidden_nodes: Number of neurons in layer
		:param activation: Activation function for the layer
		:param mini_batch: number of training sample
		:param max_epoch_without_improvement: how many iterations should be done after best performance is 
			reached to probe the rmsd behaviour. 
		:param early_stop: if True, stops the iterations when there is no improvements anymore, 
			makes probes the `max_epoch_without_improvement` following iterations before stopping.
		"""
		self.hidden_nodes = hidden_nodes
		self.activation_name = activation
		self.train_history = []
		self.corruption = corruption
		
		self.activation_fct = eval("act.{}".format(self.activation_name.lower()))
		self.activation_fct_prime = eval("act.{}_prime".format(self.activation_name.lower()))
	
	def xentropy_cost(self, theta, data):
		
		m = data.shape[1]

		# Unroll theta
		self.weights, self.inverse_biases, self.biases = self._unroll(theta)

		# Forward passes
		if not self.corruption is None:
			cdata = processing.corrupt(self, data, self.corruption)
		else:
			cdata = data
		
		ch = self.round_feedforward(cdata)
		#if not self.corruption is None:
		#	h = self.feedforward(data)
		hn = self.output
			
		# Compute the gradients:
		# http://neuralnetworksanddeeplearning.com/chap3.html for details
		
		# First: bottom to top
		dEda = ch - data
		dEdvb = np.mean(dEda, axis=0)

		# Second: top to bottom
		dEda = (dEda).dot(self.weights) * (hn * (1. - hn))
		dEdhb = np.mean(dEda, axis=0)
		
		dEdw = ((dEda.T).dot(data).T + self.regularisation * self.weights) / m
		grad = self._roll(dEdw, dEdvb, dEdhb)

		# Computes the cross-entropy
		self.current_cost_value = utils.cross_entropy(data, ch)
		
		# Returns the gradient as a vector.
		return self.current_cost_value, grad
	
	def l2_cost(self, theta, data):
		"""
		Computes the L2 cost and gradient
		http://neuralnetworksanddeeplearning.com/chap3.html for details
		"""

		m = data.shape[1]
			

		# Unroll theta
		self.weights, self.inverse_biases, self.biases = self._unroll(theta)

		# Forward passes
		
		if self.corruption is not None:
			cdata = processing.corrupt(self, data, self.corruption)
			ch = self.round_feedforward(cdata)
		else:
			cdata = data

		h, activation_last = self.round_feedforward(data, return_activation=True)
		if self.corruption is None:
			ch = h
			
		# Back-propagation
		prime = self.activation_fct_prime(activation_last)
		deltaL = (ch - data) * prime
		dEdvb = np.mean(deltaL, axis=0)
		
		prime = self.activation_fct_prime(self.activation)
		
		deltal = np.multiply((np.dot(deltaL, self.weights)), prime)
		deltal = np.array(deltal)
		dEdhb = np.mean(deltal, axis=0) 
		
		dEdw = (((deltal.T).dot(data)).T + self.regularisation * self.weights) / m

		grad = self._roll(dEdw, dEdvb, dEdhb)
		
		self.current_cost_value = np.sum((h - data) ** 2) / (2 * m) + ((self.regularisation / 2) * np.abs(self.weights)**2).sum()
					
		
		
		# Returns the gradient as a vector.
		return self.current_cost_value, grad
	
	def _log(self, *args, **kwargs):
		"""
		Used as a callback function for the minization algorithm
		"""
		self.train_history.append(self.current_cost_value)
		
	
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
		"""
		self.weights2 = 4 * np.random.uniform(
					low=-np.sqrt(6. / (numdims + self.hidden_nodes)),
					high=np.sqrt(6. / (numdims + self.hidden_nodes)),
					size=(numdims, self.hidden_nodes))
	
		self.weights3 = 2. * np.random.uniform(
					low=-np.sqrt(1. / (numdims)),
					high=np.sqrt(1. / (numdims)),
					size=(numdims, self.hidden_nodes))
		
		#print 'WEIGHTS:', np.amin(self.weights), np.amax(self.weights), np.mean(self.weights), np.std(self.weights)
		"""
		
		self.inverse_biases = np.zeros(numdims)
		self.biases = np.zeros(self.hidden_nodes)
		time_start = datetime.now()
	
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

