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
		
	def feedforward(self, data, mem=False, debug=False):	
		m_input = data 
		if debug:
			print np.shape(m_input), '< data'
			print np.shape(self.weights), '< weights'
			print np.shape(self.hidden_biases), '< bias'
			print np.shape(m_input), np.shape(self.weights)
		
		m_output = self.compute_layer(m_input)
		
		if mem: self.activation = m_output
		
		if(self.hidden_type == "SIGMOID"):
			m_output = utils.sigmoid(m_output)
		elif(self.hidden_type == "LINEAR"):
			pass # Nothing to do here
		else:
			raise NotImplemented("Unrecogonised hidden type")
			
		return m_output
	
	def feedforward_memory(self, data):
		m_output = self.feedforward(data, mem = True)
		self.output = m_output
		self.input = data
		
		return m_output
	
	def plot_train_history(self):
		import pylab as plt
		plt.figure()
		plt.plot(np.arange(np.size(self.train_history)), self.train_history, lw=2)
		plt.show()
