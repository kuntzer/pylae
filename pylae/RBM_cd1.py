import numpy as np
import cPickle as pickle 
import utils as u
import layer

class RBM(layer.AE_layer):
	def __init__(self, hidden_nodes, visible_type, hidden_type, mini_batch, iterations, 
				max_epoch_without_improvement=50, early_stop=True):
		"""
		:param hidden_nodes: Number of neurons in layer
		:param visible_type: `SIGMOID` or `LINEAR`, linear for in-/outputs
		:param hidden_type: `SIGMOID` or `LINEAR`, sigmoid for hidden layers
		:param mini_batch: number of training sample
		:param learning_rate: learning rate, set to 0.01 input / 0.1 afterwards
		:param momentum_rate: 0.9 See code --> docstring need todo
		:param weight_decay: NOT IMPLEMENTED YET
		"""
		self.hidden_nodes = hidden_nodes
		self.visible_type = visible_type
		self.hidden_type = hidden_type
		self.mini_batch = mini_batch
		self.iterations = iterations
		self.max_epoch_without_improvement = max_epoch_without_improvement
		self.early_stop = early_stop
	
	def train(self, data_train, initialmomentum, finalmomentum, learn_rate_w,
			learn_rate_visb, learn_rate_hidb, weightcost):
		
		learning_rate = learn_rate_w
		
		self.weightcost = weightcost
		
		N = np.shape(data_train)[0]
		num_visible = np.shape(data_train)[1]

		total_batches = np.int(np.ceil(N/self.mini_batch))

		m_visible_biases = np.zeros(num_visible)
		m_hidden_biases = np.zeros(self.hidden_nodes)

		m_weights = np.random.randn(num_visible, self.hidden_nodes) * 0.01
		momentum_weights = np.zeros_like(m_weights)
		momentum_visible_biases = np.zeros_like(m_visible_biases)
		momentum_hidden_biases = np.zeros_like(m_hidden_biases)
		
		momentum_rate = finalmomentum
		
		best_rmsd = None
		rmsd_logger=[]
		
		for i in range(self.iterations):
			start = (i % total_batches) * self.mini_batch
			end = start + self.mini_batch			
			if end >= N : end = N - 1
			
			#print start, end
			
			batch = data_train[start:end]
			batch = batch
			
			weights, visible_biases, hidden_biases = self._CD1(batch, 
				m_weights, m_visible_biases, m_hidden_biases)
			
			momentum_weights = momentum_rate * momentum_weights + weights
			momentum_visible_biases = momentum_rate * momentum_visible_biases + visible_biases
			momentum_hidden_biases = momentum_rate * momentum_hidden_biases + hidden_biases
			
			m_weights += momentum_weights * learning_rate
			m_visible_biases += momentum_visible_biases * learning_rate
			m_hidden_biases += momentum_hidden_biases * learning_rate
			
			# Reconstruction error, no sampling done, using raw probability
			
			hidden_state = u.sigmoid(np.dot(data_train, m_weights) \
										+ np.tile(hidden_biases, (N,1)))
			
			reconstruction = np.dot(hidden_state, m_weights.T) +  \
										+ np.tile(visible_biases, (N,1))

			if self.visible_type == "SIGMOID":
				reconstruction = u.sigmoid(reconstruction)
				
			##################################################
				
			err = (data_train-reconstruction)*(data_train-reconstruction)
			rmsd = np.sqrt(np.mean(err*err))
			rmsd_logger.append(rmsd)
			print "Epoch %4d/%4d, RMS deviation = %7.4f" % (i + 1, self.iterations, rmsd)
			
			if self.early_stop:			
				if best_rmsd is None or (rmsd - best_rmsd) / best_rmsd < -1e-3 :
					best_weights = m_weights
					best_hidden_biases = hidden_biases
					best_visible_biases = visible_biases
					best_rmsd = rmsd
					iter_since_best = 0
				else :
					iter_since_best += 1
					
				if iter_since_best >= self.max_epoch_without_improvement :
					print "Early stop -- best epoch %d / %d, RMS deviation = %7.4f" % (
						i + 1 - iter_since_best, self.iterations, best_rmsd)
					break
				
			else: 
				best_weights = m_weights
				best_hidden_biases = hidden_biases
				best_visible_biases = visible_biases
				best_rmsd = rmsd
				iter_since_best = 0
			
			#print ' rmsd = ', rmsd
		self.weights = best_weights
		self.hidden_biases = best_hidden_biases
		self.visible_biases = best_visible_biases
		rmsd_history = np.asarray(rmsd_logger)
		if iter_since_best > 0 :
			self.train_history = rmsd_history[:-1*iter_since_best]
		else :
			self.train_history = rmsd_history
			#exit()
			
	
	def _CD1(self, visible_data, weights, visible_bias, hidden_bias):
		N = np.shape(visible_data)[0]
		# Positive phase
		visible_state = visible_data
		
		if self.visible_type == "SIGMOID" :
			visible_state = self._samplebinary(visible_state)
		elif self.visible_type == "LINEAR" :
			visible_state = self._add_gaussian_noise(visible_state);

		
		nw = np.dot(visible_state, weights) + np.tile(hidden_bias, (N, 1))
		if self.hidden_type == "SIGMOID":
			hidden_probability = u.sigmoid(nw) 
			hidden_state = self._samplebinary(hidden_probability)
		elif self.hidden_type == "LINEAR":
			hidden_state = self._add_gaussian_noise(nw)
			
		gradient1 = self._gradient_weights(visible_state, hidden_state, weights)
		visible_biases1 = self._gradient_biases(visible_state, visible_bias)
		hidden_biases1 = self._gradient_biases(hidden_state, hidden_bias)

		# Negative phase
		# Skip sampling as well...
		visible_state = np.dot(hidden_state, weights.T) + np.tile(visible_bias, (N, 1))
		
		if self.visible_type == "SIGMOID":
			visible_state = u.sigmoid(visible_state)
			#visible_probability = u.sigmoid(visible_state)
			#visible_state = self._samplebinary(visible_probability)
			
		# skip sampling here
		
		nw = np.dot(visible_state, weights) + np.tile(hidden_bias, (N, 1))
		if self.hidden_type == "SIGMOID":
			hidden_probability = hidden_probability = u.sigmoid(nw) 
			hidden_state = hidden_probability
		elif self.hidden_type == "LINEAR" :
			hidden_state = nw

		gradient2 = self._gradient_weights(visible_state, hidden_state, weights)
		visible_biases2 = self._gradient_biases(visible_state, visible_bias)
		hidden_biases2 = self._gradient_biases(hidden_state, hidden_bias)
		
		# gradients
		weights = gradient1 - gradient2;
		visible_biases = visible_biases1 - visible_biases2;
		hidden_biases= hidden_biases1 - hidden_biases2;

		return weights, visible_biases, hidden_biases
			
	def _samplebinary(self, data):

		output = np.zeros_like(data)
		N = np.size(data)
		r = np.random.rand(N)
		r = r.reshape(np.shape(data))

		output[data > r] = 1

		return output
	
	def _add_gaussian_noise(self, m):
		m += np.random.randn(np.size(m)).reshape(np.shape(m))
		return m
		

	def _gradient_weights(self, visible_state, hidden_state, weights):
		#print np.shape(visible_state), np.shape(hidden_state)
		d_G_by_rbm_w = np.zeros([np.shape(visible_state)[1], np.shape(hidden_state)[1]])
		m = np.shape(visible_state)[0]
		
		for i in range(m):
# 			print np.shape(visible_state[i])
			hs = hidden_state[i].reshape(1, np.shape(hidden_state[i])[0])
			vs = visible_state[i].reshape(np.shape(visible_state[i])[0], 1)
			
			d_G_by_rbm_w += np.dot(vs, hs)

		d_G_by_rbm_w /= m
		
		# weight penalty
# 		print np.shape(d_G_by_rbm_w)
# 		print np.shape(weights)
# 		print np.shape(self.weightcost*weights)
		d_G_by_rbm_w -= self.weightcost*weights

		return d_G_by_rbm_w
	
	def _gradient_biases(self, state, bias):
		N = np.shape(state)[0]
		
		if self.visible_type == "SIGMOID":
			return np.mean(state, axis=0)
		
		elif self.visible_type == "LINEAR":
			d_G_by_vbias = np.zeros_like(bias)

		for i in range(N):
			d_G_by_vbias += state[i] - bias; # deriv of -E, not E
		d_G_by_vbias /= N;

		return d_G_by_vbias;
