import numpy as np
import utils

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
		
	def compute_layer(self, m_input):
		return np.dot(m_input, self.weights) + self.hidden_biases
	
	def compute_visible(self, m_input):
		return np.dot(m_input, self.weights.T) + self.visible_biases
	
	def activate(self, activation):
		
		if(self.visible_type == "SIGMOID"):
			m_output = utils.sigmoid(activation)
		elif(self.visible_type == "RELU"):
			m_output = utils.relu(activation)
		elif(self.visible_type == "LEAKY_RELU"):
			m_output = utils.leaky_relu(activation)
		elif(self.visible_type == "LINEAR"):
			m_output = activation
		else:
			raise NotImplemented("Unrecogonised hidden type")
	
		return m_output
	
	def prime_activate(self, activation):
		if(self.hidden_type == "SIGMOID"):
			prime = utils.sigmoid_prime(activation)
		elif(self.visible_type == "RELU"):
			prime = utils.relu_prime(activation)
		elif(self.visible_type == "LEAKY_RELU"):
			prime = utils.leaky_relu_prime(activation)
		elif(self.hidden_type == "LINEAR"):
			prime = activation
		else:
			raise NotImplemented("Unrecogonised hidden type")
			
		return prime
	
	def full_feedforward(self, data, return_activation=False, dropout=None, return_hidden=False):
		hidden_data = self.feedforward_memory(data, dropout=dropout)
		activation = self.compute_visible(hidden_data)
		
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
			print np.shape(self.hidden_biases), '< bias'
			print np.shape(m_input), np.shape(self.weights)
		
		activation = self.compute_layer(m_input)
		
		m_output = self.activate(activation)

		if dropout is not None:
			print 'dropout!'
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
	
	def plot_train_history(self):
		import pylab as plt
		plt.figure()
		plt.plot(np.arange(np.size(self.train_history)), self.train_history, lw=2)
		plt.show()
