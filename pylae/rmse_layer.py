import numpy as np
import layer
import scipy.optimize
import utils as u
import pylab as plt

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
		
		# Unroll theta
		self.weights, self.visible_biases, self.hidden_biases = self._unroll(theta)

		# Forward passes
		h, activation_last = self.full_feedforward(data.T, return_activation=True)
		h = h.T

			
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
		deltaL = (h - data) * u.sigmoid_prime(activation_last.T)
		dEdvb = np.mean(deltaL, axis=1)
		deltal = np.dot((self.weights.T), deltaL) * u.sigmoid_prime(self.activation.T)
		dEdhb = np.mean(deltal, axis=1)
		
		dEdw = deltal.dot(data.T).T / m + lambda_ * self.weights / m

		
		grad = self._roll(dEdw, dEdvb, dEdhb)
		
		cost = np.sum((h - data) ** 2) / (2 * m) + ((lambda_ / 2) * np.abs(self.weights)**2).sum()
	
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
	
	def _corrupt(self, data):
		
		if type(self.corruption) == float:
			cdata = np.random.binomial(size=data.shape, n=1, p=1.-self.corruption) * data
		elif np.shape(np.asarray(self.corruption).T) == np.shape(data):
			cdata = self.corruption.T
		else:
			print np.amin(data), np.amax(data), np.mean(data), np.std(data)
			if self.data_std is not None and self.data_norm is not None:
				scales = np.random.uniform(low=self.corruption[0], high=self.corruption[1], size=data.shape[1])
				
				data = u.unnormalise(data, self.data_norm[0], self.data_norm[1])
				data = u.unstandardize(data, self.data_std[0], self.data_std[1])
				
				noise_maps = [np.random.normal(scale=sig, size=data.shape[0]) for sig in scales]
				noise_maps = np.asarray(noise_maps)
				
				cdata = data + noise_maps.T
				
				cdata, _, _ = u.standardize(cdata, self.data_std[0], self.data_std[1])
				cdata, _, _ = u.normalise(cdata, self.data_norm[0], self.data_norm[1])
				
				# Just making sure we're not out of bounds:
				min_thr = 1e-6
				max_thr = 0.99999
				
				print 'N/C:', (cdata < min_thr).sum(), (cdata > max_thr).sum()
				cdata[cdata < min_thr] = min_thr
				cdata[cdata > max_thr] = max_thr
				
				print np.amin(cdata), np.amax(cdata), np.mean(cdata), np.std(cdata)
			else:
				raise RuntimeError("Can't normalise the data. You must provide the normalisation and standardisation values. Giving up.")
		#print np.amin(data), np.amax(data)
		#print np.amin(cdata), np.amax(cdata)
		return cdata
	
	def train(self, data, data_std=None, data_norm=None, method='L-BFGS-B', verbose=True, return_info=False, **kwargs):
		# TODO: deal with minibatches!
		_, numdims = np.shape(data)
		self.data_std = data_std
		self.data_norm = data_norm
		#N = self.mini_batch
		self.visible_dims = numdims
		
		self.weights = 0.001 * np.random.randn(numdims, self.hidden_nodes)
		
		# This apparently is better, but definitely noisier (or plain worse)!
		self.weights2 = 4 * np.random.uniform(
					low=-np.sqrt(6. / (numdims + self.hidden_nodes)),
					high=np.sqrt(6. / (numdims + self.hidden_nodes)),
					size=(numdims, self.hidden_nodes))
		
		self.visible_biases = np.zeros(numdims)
		self.hidden_biases = np.zeros(self.hidden_nodes)
		
		theta = self._roll(self.weights, self.visible_biases, self.hidden_biases)
	
		data = data.T
		
		###########################################################################################
		# Optimisation of the weights and biases according to the cost function
		J = lambda x: self.cost(x, data, log_cost=True, **kwargs)
		
		options_ = {'maxiter': self.iterations, 'disp': verbose, 'ftol' : 10. * np.finfo(float).eps, 'gtol': 1e-9}
		result = scipy.optimize.minimize(J, theta, method=method, jac=True, options=options_)
		opt_theta = result.x
		
		###########################################################################################
		# Unroll the state vector and saves it to self.	
		self.weights, self.visible_biases, self.hidden_biases = self._unroll(opt_theta)
		
		if verbose: print result
				
		# If the user requests the informations about the minimisation...
		if return_info: return result
	
		
