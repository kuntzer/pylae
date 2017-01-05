import numpy as np

class AE_layer():
	def __init__(self, hidden_nodes, visible_type, hidden_type, mini_batch, iterations, 
				max_epoch_without_improvement=50, early_stop=True):
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
		
		self.act_fct = eval("act.{}".format(self.activation.lower()))
		self.act_fct_prime = eval("act.{}_prime".format(self.activation.lower()))

	def compute_layer(self, m_input):
		return np.dot(m_input, self.weights) + self.biases
	
	def compute_inverse(self, m_input):
		return np.dot(m_input, self.weights.T) + self.inverse_biases
	
	def activate(self, activation):
		m_output = self.activation_fct(activation)
		return m_output

	def round_feedforward(self, data, return_activation=False, dropout=None, return_hidden=False):
		hidden_data = self.feedforward_memory(data, dropout=dropout)
		activation = self.compute_inverse(hidden_data)
		m_output = self.activate(activation)
			
		#if dropout is not None:
			#print 'dropout!'
			#dropouts = np.random.binomial(1, dropout, size=m_output.shape) * (1. / (1. - dropout))
			#m_output *= dropouts
			#activation *= dropouts
			
		if return_activation and return_hidden:
			return m_output, activation, hidden_data
		elif return_activation:
			return m_output, activation
		elif return_hidden:
			return hidden_data
		
		return m_output
		
	def feedforward(self, data, mem=False, debug=False, dropout=None):	
		m_input = data 
		if debug:
			print np.shape(m_input), '< data'
			print np.shape(self.weights), '< weights'
			print np.shape(self.biases), '< bias'
			print np.shape(m_input), np.shape(self.weights)
		
		activation = self.compute_layer(m_input)
		
		m_output = self.activate(activation)

		if dropout is not None:
			print 'dropout!'
			raise ValueError("Should be checked before use!")
			dropouts = np.random.binomial(1, 1.-dropout, size=m_output.shape) * (1. / (1. - dropout))
			m_output *= dropouts
			activation *= dropouts

		if mem: self.activation = activation
		return m_output
	
	def feedforward_memory(self, data, dropout=None):
		m_output = self.feedforward(data, mem = True, dropout=dropout)
		self.output = m_output
		self.input = data
		
		return m_output
	
	def _roll(self, weights, inverse_biases, biases):
		return np.concatenate([weights.ravel(), inverse_biases, biases])
	
	def _unroll(self, theta):
		
		nw = np.size(self.weights)
		nvb = np.size(self.inverse_biases)
		
		weights = theta[:nw]
		inverse_biases = theta[nw:nw+nvb]
		biases = theta[nw+nvb:]
		
		weights = weights.reshape([self.visible_dims, self.hidden_nodes])
		
		return weights, inverse_biases, biases

