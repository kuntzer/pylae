import numpy as np
import scipy.optimize

import layer
import processing
import utils as u

class Layer(layer.AE_layer):
	def __init__(self, hidden_nodes, visible_type, hidden_type, mini_batch, iterations, 
				corruption, max_epoch_without_improvement=50, early_stop=True):
		"""
		:param hidden_nodes: Number of neurons in layer
		:param visible_type: `SIGMOID` or `LINEAR`, linear for in-/outputs
		:param hidden_type: `SIGMOID` or `LINEAR`, sigmoid for hidden layers
		:param mini_batch: number of training sample
		:param max_epoch_without_improvement: how many iterations should be done after best performance is 
			reached to probe the rmsd behaviour. 
		:param early_stop: if True, stops the iterations when there is no improvements anymore, 
			makes probes the `max_epoch_without_improvement` following iterations before stopping.
		"""
		self.hidden_nodes = hidden_nodes
		self.visible_type = visible_type
		self.hidden_type = hidden_type
		self.mini_batch = mini_batch
		self.iterations = iterations
		self.max_epoch_without_improvement = max_epoch_without_improvement
		self.early_stop = early_stop
		self.train_history = []
		self.corruption = corruption
		
	
	def cost(self, theta, data, log_cost=False, **params):

		if not 'regularisation' in params:
			lambda_ = 0.
		else:
			lambda_ = params['regularisation']

		m = data.shape[1]
		
		# Unroll theta
		self.weights, self.visible_biases, self.hidden_biases = self._unroll(theta)

		# Forward passes
		if not self.corruption is None:
			cdata = processing.corrupt(self, data, self.corruption).T
		else:
			cdata = data.T
			
		h = self.full_feedforward(cdata).T
		#if not self.corruption is None:
		#	hn = self#self.feedforward(data.T)
		#else:
		#	pass
		hd = self.feedforward(data.T)
		hn = self.output
			
		# Compute the gradients:
		# http://neuralnetworksanddeeplearning.com/chap3.html for details
		
		# First: bottom to top
		dEda = h - data
		dEdvb = np.mean(dEda, axis=1)
		#Second: top to bottom
		
		
		dEda = (self.weights.T).dot(dEda) * (hn * (1. - hn)).T
		dEdhb = np.mean(dEda, axis=1)
		
		dEdw = dEda.dot(data.T) / m + lambda_ * self.weights.T / m
		dEdw = dEdw.T
		grad = self._roll(dEdw, dEdvb, dEdhb)

		# Computes the cross-entropy
		cost = - np.sum(data * np.log(h) + (1. - data) * np.log(1. - h), axis=0) 
		cost = np.mean(cost)
				
		if log_cost:
			self.train_history.append(cost)
		
		# Returns the gradient as a vector.
		return cost, grad

	
	def _roll(self, weights, visible_biases, hidden_biases):
		return np.concatenate([weights.ravel(), visible_biases, hidden_biases])
	
	def _unroll(self, theta):
		
		nw = np.size(self.weights)
		nvb = np.size(self.visible_biases)
		#nhb = np.size(self.hidden_biases)
		
		weights = theta[:nw]
		visible_biases = theta[nw:nw+nvb]
		hidden_biases = theta[nw+nvb:]
		
		weights = weights.reshape([self.visible_dims, self.hidden_nodes])
		
		return weights, visible_biases, hidden_biases
	
	def train(self, data, data_std=[0.,1.], data_norm=[0., 1.], method='L-BFGS-B', verbose=True, return_info=False, weight=0.1, **kwargs):
		# TODO: deal with minibatches!
		
		_, numdims = np.shape(data)
		self.data_std = data_std
		self.data_norm = data_norm
		#N = self.mini_batch
		self.visible_dims = numdims
		
		self.weights = weight * np.random.randn(numdims, self.hidden_nodes)
		
		# This apparently is better, but definitely noisier (or plain worse)!
		self.weights2 = 4 * np.random.uniform(
					low=-np.sqrt(6. / (numdims + self.hidden_nodes)),
					high=np.sqrt(6. / (numdims + self.hidden_nodes)),
					size=(numdims, self.hidden_nodes))
	
		self.weights3 = 2. * np.random.uniform(
					low=-np.sqrt(1. / (numdims)),
					high=np.sqrt(1. / (numdims)),
					size=(numdims, self.hidden_nodes))
		
		#print 'WEIGHTS:', np.amin(self.weights), np.amax(self.weights), np.mean(self.weights), np.std(self.weights)
		
		self.visible_biases = np.zeros(numdims)
		self.hidden_biases = np.zeros(self.hidden_nodes)
		
		theta = self._roll(self.weights, self.visible_biases, self.hidden_biases)
	
		data = data.T
		
		###########################################################################################
		# Optimisation of the weights and biases according to the cost function
		J = lambda x: self.cost(x, data, log_cost=True, **kwargs)

		options_ = {'maxiter': self.iterations, 'disp': verbose, 'ftol' : 10. * np.finfo(float).eps}
		result = scipy.optimize.minimize(J, theta, method=method, jac=True, options=options_)
		opt_theta = result.x
		
		###########################################################################################
		# Unroll the state vector and saves it to self.	
		self.weights, self.visible_biases, self.hidden_biases = self._unroll(opt_theta)
		
		if verbose: print result
				
		# If the user requests the informations about the minimisation...
		if return_info: return result
	
		
