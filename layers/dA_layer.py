import numpy as np
import scipy.optimize

import layer
from .. import processing
from .. import act 
from .. import utils

class Layer(layer.AE_layer):
	def __init__(self, hidden_nodes, activation, corruption, verbose=True):
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
		self.verbose = verbose
		
		self.activation_fct = eval("act.{}".format(self.activation_name.lower()))
	
	def cost(self, theta, data, log_cost=False):

		if self.mini_batch <= 0:
			batch = data
		else:
			ids_batch = self._select_mini_batch()
			batch = data[:,ids_batch]

		m = batch.shape[1]
		
		# Unroll theta
		self.weights, self.inverse_biases, self.biases = self._unroll(theta)

		# Forward passes
		if not self.corruption is None:
			cdata = processing.corrupt(self, batch, self.corruption).T
		else:
			cdata = batch.T
			
		h = self.round_feedforward(cdata).T
		hn = self.output
			
		# Compute the gradients:
		# http://neuralnetworksanddeeplearning.com/chap3.html for details
		
		# First: bottom to top
		dEda = h - batch
		dEdvb = np.mean(dEda, axis=1)

		# Second: top to bottom
		dEda = (self.weights.T).dot(dEda) * (hn * (1. - hn)).T
		dEdhb = np.mean(dEda, axis=1)
		
		dEdw = (batch.dot(dEda.T) + self.regularisation * self.weights) / m
		grad = self._roll(dEdw, dEdvb, dEdhb)

		# Computes the cross-entropy
		cost = utils.cross_entropy(batch.T, h.T)
		
		if log_cost:
			self.train_history.append(cost)
		
		# Returns the gradient as a vector.
		return cost, grad
	
	def _new_epoch(self):
		self.mini_batch_ids = np.ones(self.Ndata)
		
	def _select_mini_batch(self):
		
		if np.sum(self.mini_batch_ids) <= 0:
			self._new_epoch()
		
			if self.verbose: 
				print "A new epoch has started"
				
		if np.sum(self.mini_batch_ids) < self.mini_batch:
			b = int(self.mini_batch_ids.sum())
		else:
			b = self.mini_batch
	
		aids = np.where(self.mini_batch_ids == 1)[0]
		avail_ids = np.arange(self.Ndata)[aids]
		ids_batch = np.random.choice(avail_ids, b, replace=False)
		
		self.mini_batch_ids[ids_batch] = 0

		return np.arange(self.Ndata)[ids_batch]
	
	def train(self, data, iterations, mini_batch=0, regularisation=0, method='L-BFGS-B', weight=0.1, **kwargs):
		"""
		Pre-training of a dA layer with cross-entropy (hard coded -- at least for now)
		
		:param data: The data to be used
		:param iterations: the number of epochs (i.e. nb of loops of mini_batch)
		:param mini_batch: number of samples to use per batch
		:param regularisation: the multipler of the L2 regularisation
		:param method: which method of `scipy.optimize.minimize` to use
		:pram weights: the standard deviation of the zero-centred weights (biases are initialised to 0)
		
		all remaining `kwargs` are passed to `scipy.optimize.minimize`.
		"""
		
		self.Ndata, numdims = np.shape(data)
		self.mini_batch = mini_batch
		self.mini_batch_ids = self._new_epoch()
		
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
		theta = self._roll(self.weights, self.inverse_biases, self.biases)
	
		data = data.T
		
		###########################################################################################
		# Optimisation of the weights and biases according to the cost function
		J = lambda x: self.cost(x, data, log_cost=True)

		options_ = {'maxiter': self.iterations, 'disp': self.verbose}
		# We overwrite these options with any user-specified kwargs:
		options_.update(kwargs)
		
		result = scipy.optimize.minimize(J, theta, method=method, jac=True, options=options_)
		opt_theta = result.x
		
		###########################################################################################
		# Unroll the state vector and saves it to self.	
		self.weights, self.inverse_biases, self.biases = self._unroll(opt_theta)
		
		if self.verbose: print result	
		
