import numpy as np
import scipy.optimize
import pylab as plt

import layer
import utils as u
import processing

class Layer(layer.AE_layer):
	def __init__(self, hidden_nodes, visible_type, hidden_type, mini_batch, iterations, 
				corruption, max_epoch_without_improvement=50, early_stop=True, **kwargs):
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

		m = data.shape[1]
		if not 'regularisation' in params:
			lambda_ = 0.
		else:
			lambda_ = params['regularisation']
			
		if not 'show' in params:
			show = False
		else:
			show = params['show']
			
		if not 'dropout' in params:
			dropout = None
		else:
			dropout = params['dropout']
			
		if not 'sparsity' in params:
			sparsity = None
		else:
			sparsity = params['sparsity']
		"""
		if not ('X' in params and 'Y' in params and 'E1' in params and 'E2' in params and 'R2' in params):
			raise RuntimeError("Need the X, Y arrays and E1, E2, R2!")
		else:
			X = params['X'].T
			Y = params['Y'].T
			E1 = params['E1'].T
			E2 = params['E2'].T
			R2 = params['R2'].T
			F = params['F'].T
		"""
		
		if self.corruption is not None:
			cdata = processing.corrupt(self, data, self.corruption)
		else:
			cdata = data

		# Unroll theta
		self.weights, self.visible_biases, self.hidden_biases = self._unroll(theta)

		# Forward passes
		ch, nactivation_last = self.full_feedforward(cdata.T, return_activation=True, dropout=dropout)
		h, activation_last, hidden_data = self.full_feedforward(data.T, return_activation=True, dropout=dropout, return_hidden=True)
		ch = ch.T # TODO: clear this out! Really weird
		h = h.T
		
		rhohat = np.mean(hidden_data, axis=0)

		#exit()
		"""
		plt.figure()
		plt.subplot(221)
		plt.imshow((cdata.T)[0].reshape(24,24), interpolation="None")
		plt.title('cdata')
		plt.subplot(222)
		plt.imshow((data.T)[0].reshape(24,24), interpolation="None")
		plt.title('data')
		plt.subplot(223)
		plt.imshow((ch.T)[0].reshape(24,24), interpolation="None")
		plt.title('ch')
		plt.subplot(224)
		plt.imshow((h.T)[0].reshape(24,24), interpolation="None")
		plt.title('h')
		plt.show()
		"""
		# Compute the gradients:
		# http://neuralnetworksanddeeplearning.com/chap3.html for details
		"""

		hn = self.output
		
		# First: bottom to top
		sI = np.sum((X*X + Y*Y) * h, axis = 0)
		dR = (np.sum((X * X + Y * Y) * h, axis=0) - R2)
		dE1 = (np.sum((X * X + Y * Y) * h, axis=0) - E1) 
		dE2 = (np.sum((X * Y) * h, axis=0) - E2) 
		dF = (np.sum(h, axis=0) - F) 
		dM = np.sum((X + Y) * h, axis=0) - (np.sum((X + Y) * data, axis=0))
		dEda = (X*X + Y*Y) * dR + (X*X-Y*Y) * dE1 + X * Y * dE2 + (X + Y) * dM
		dEda /= sI
		dEda += dF
		#print dEda
		#print X.shape 
		#print dR.shape
		#dEda /= 24*24
		#print dEda
		#exit()
		dEda *= 2.
		dEdvb = np.mean(dEda, axis=1)
		
		#Second: top to bottom
		dEda = (self.weights.T).dot(dEda) * (hn * (1. - hn)).T
		dEdhb = np.mean(dEda, axis=1)
		
		dEdw = dEda.dot(data.T) / m
		dEdw = dEdw.T
		grad = self._roll(dEdw, dEdvb, dEdhb)
		# Compute the cost
		#cost = #(np.sum((X * X - Y * Y) * h, axis=0) - E1) ** 2 \
			# + (2. * np.sum((X * Y) * h, axis=0) - E2) ** 2 + \
		cost = (np.sum((X * X + Y * Y) * h, axis=0) - R2) ** 2
		cost = np.mean(np.sqrt(cost))
		"""
		# Back-propagation
		#delta = -(data - h)
		#dEdvb = np.mean(delta, axis=1)
		#delta = np.dot(delta, self.weights)# (self.weights.T).dot(delta)
		prime = self.prime_activate(activation_last.T)
		deltaL = (ch - data) * prime
		dEdvb = np.mean(deltaL, axis=1)
		
		prime = self.prime_activate(self.activation.T)
		
		if sparsity is not None:
			spars_cost = sparsity[0] * np.sum(u.KL_divergence(sparsity[1], rhohat))
			spars_grad = sparsity[0] * u.KL_prime(sparsity[1], rhohat)
			spars_grad = np.matrix(spars_grad).T
			#spars_grad = np.tile(spars_grad, m).reshape(m,self.hidden_nodes).T
			#print rhohat.mean(), 'cost:', spars_cost, '<<<<<<<<<<<<<<<'
		else:
			spars_cost = 0.
			spars_grad = 0.
		deltal = np.multiply((np.dot((self.weights.T), deltaL) + spars_grad), prime)
		deltal = np.array(deltal)
		dEdhb = np.mean(deltal, axis=1) 
		
		dEdw = deltal.dot(data.T).T / m + lambda_ * self.weights / m

		grad = self._roll(dEdw, dEdvb, dEdhb)
		
		cost = np.sum((h - data) ** 2) / (2 * m) + ((lambda_ / 2) * np.abs(self.weights)**2).sum() \
				+ spars_cost
		#print 'tot cost', cost
	
		if show:
			rid = np.random.uniform(0, m)
			#rid = 50
			size = np.int(np.sqrt(h[:,rid].size))
			plt.figure()
			plt.subplot(131)
			plt.imshow(h[:,rid].reshape(size, size), interpolation='None')
			plt.subplot(132)
			plt.imshow(self.input[rid].reshape(size, size), interpolation='None')
			plt.subplot(133)
			plt.imshow((h[:,rid]-self.input[rid]).reshape(size, size), interpolation='None')
			plt.show()	
		
					
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
	
	def train(self, data, data_std=[0.,1.], data_norm=[0., 1.], method='L-BFGS-B', verbose=True, return_info=False, weight=0.001, **kwargs):
		# TODO: deal with minibatches!
		_, numdims = np.shape(data)
		self.data_std = data_std
		self.data_norm = data_norm
		#N = self.mini_batch
		self.visible_dims = numdims
		
		self.weights = weight * np.random.randn(numdims, self.hidden_nodes)
		
		# This apparently is better, but definitely noisier (or plain worse)!
		self.weights2 = np.random.uniform(
					low=-np.sqrt(6. / (numdims + self.hidden_nodes)),
					high=np.sqrt(6. / (numdims + self.hidden_nodes)),
					size=(numdims, self.hidden_nodes))
		
		self.visible_biases = np.zeros(numdims)
		self.hidden_biases = np.zeros(self.hidden_nodes)
		
		theta = self._roll(self.weights, self.visible_biases, self.hidden_biases)
	
		data = data.T
		
		###########################################################################################
		# Optimisation of the weights and biases according to the cost function
		if self.iterations:
			J = lambda x: self.cost(x, data, log_cost=True, **kwargs)
			
			options_ = {'maxiter': self.iterations, 'disp': verbose, 'ftol' : 10. * np.finfo(float).eps, 'gtol': 1e-9}
			result = scipy.optimize.minimize(J, theta, method=method, jac=True, options=options_)
			opt_theta = result.x
		else:
			opt_theta = theta
			result = None
		
		###########################################################################################
		# Unroll the state vector and saves it to self.	
		self.weights, self.visible_biases, self.hidden_biases = self._unroll(opt_theta)
		
		if verbose: print result
				
		# If the user requests the informations about the minimisation...
		if return_info: return result
	
		
