import numpy as np
import utils
import copy
import RBM

class AutoEncoder():
	
	def __init__(self, name='', verbose=False):
		self.name = name
		self.is_pretrained = False
		self.is_trained = False
		self.verbose = verbose
		
	def set_autencoder(self, encoder):
		"""
		From a list of layers `encoder`, create an autoencoder by adding in a reversed order of the
		encoder list to the encoder. Swaps the weights and biases too.
		:param encoder: A list of pre-trained RBMs in the encoder order.
		:type list:
		"""
		self.rbms = encoder
		
		encoder = copy.deepcopy(encoder)
		decoder = copy.deepcopy(encoder)
		decoder = decoder[::-1]
		self.layers = encoder + decoder
		
		# Swap the values for the decoder part
		self.mid = len(self.layers)/2
		for layer in self.layers[self.mid:] :
			layer.weights = layer.weights.T
			layer.hidden_biases, layer.visible_biases = layer.visible_biases, layer.hidden_biases
			layer.hidden_type, layer.visible_type = layer.visible_type, layer.hidden_type
			
		
	def feedforward(self, data):
		"""
		Encodes and decodes the data using the full autoencoder
		:param data: The data in the same format than the training data
		
		:returns: the fed-forward data
		"""
		for layer in self.layers :
			#print layer.hidden_nodes, '-'*30
			data = layer.feedforward_memory(data)
			
		return data
	
	def pre_train(self, data, architecture, layers_type, learn_rate={'SIGMOID':3.4e-3, 'LINEAR':3.4e-4},
			initialmomentum=0.53, finalmomentum=0.93, iterations=2000, mini_batch=100,
			regularisation=0.001, max_epoch_without_improvement=30, early_stop=True):

		"""
		
		"""
		
		assert len(architecture) + 1 == len(layers_type)
		
		# Archiving the configuration
		rbms_config = {'architecture':architecture, 'layers_type':layers_type, 'learn_rate':learn_rate,
				'initialmomentum':initialmomentum, 'finalmomentum':finalmomentum,
				'regularisation':regularisation, 'iterations':iterations, 
				'max_epoch_without_improvement':max_epoch_without_improvement,
				'early_stop':early_stop}
		self.rbms_config = rbms_config
		
		rbms = []
		for ii in range(len(architecture)):
			print "Pre-training layer %d..." % (ii + 1)
			learnr = learn_rate[layers_type[ii+1]]

			layer = RBM.RBM(architecture[ii], layers_type[ii], layers_type[ii+1], mini_batch, 
						iterations, max_epoch_without_improvement=max_epoch_without_improvement, 
						early_stop=early_stop)
			layer.train(data, learn_rate_w=learnr, learn_rate_visb=learnr, 
					learn_rate_hidb=learnr, initialmomentum=initialmomentum, 
					finalmomentum=finalmomentum,
					regularisation=regularisation)
			data = layer.feedforward(data)
	
			rbms.append(layer)
		
		self.is_pretrained = True
		self.set_autencoder(rbms)
	
	def encode(self, data):
		"""
		Encodes the data using the encoder
		:param data: The data in the same format than the training data
		
		:returns: encoded data
		"""
		for layer in self.layers[:self.mid] :
			print "encode for layer ", layer.hidden_nodes
			data = layer.feedforward(data)
			
		return data
		
	def decode(self, data):
		"""
		Decodes the data using the decoder
		:param data: encoded data
		
		:returns: decoded data
		"""
		for layer in self.layers[self.mid:] :
			print "decode for layer ", layer.hidden_nodes
			data = layer.feedforward(data)
			
		return data
	
	def backpropagation(self, data, iterations=500, learn_rate=0.13, momentum_rate=0.83, 
					max_epoch_without_improvement=30, early_stop=True):
		
		regularisation = 0.#0001
		
		if not self.is_pretrained: 
			raise RuntimeError("The autoencoder is not pre-trained.")

		
		N = np.shape(data)[0]
		momentum = [None] * len(self.layers)
		momentum_bias = [None] * len(self.layers)
		
		assert N > 0
		rmsd_logger = []

		best_rmsd = None
		iter_since_best = 0
		
		for epoch in range(iterations) :
			X = self.feedforward(data)
			
			# Start backpropagation
			
			# Initialise the accumulated derivating using square error E = 0.5*(T-Y)^2
			accum_deriv =  -(data - X); # dE/dz
			rmsd = np.sqrt(np.mean(accum_deriv*accum_deriv))
			
			# Backpropagate through the top to bottom
			for jj in range(self.mid * 2 - 1, -1, -1):
				layer = self.layers[jj]
				if layer.hidden_type == "SIGMOID" :
					# later part is error * gradient of cost function
					accum_deriv = accum_deriv * (1.0 - layer.output) * layer.output
					grad_bias = np.mean(accum_deriv, axis=0)# dJ/dB is error only		
				elif layer.hidden_type == "LINEAR" :
					grad_bias = np.mean(layer.output - layer.hidden_biases, axis=0)/N
				else :
					raise ValueError("Type of layer not recognised")

				grad = np.dot(layer.input.T, accum_deriv)/N  + regularisation * layer.weights # dz/dW partial derivative for weights
					
				
				if np.any(grad) > 1e9 or np.any(grad) is np.nan :
					raise ValueError("Weights have blown to infinite! \
						Try reducing the learning rate.")
					
				# no point accumulating gradients for the first layer
				if jj > 0:
					accum_deriv = np.dot(accum_deriv, layer.weights.T) 
					
			#TODO: Regularisation ?
			#HERE
			# SEE http://ufldl.stanford.edu/wiki/index.php/Backpropagation_Algorithm
			# ARE WE REALLY UPDATING THE BIAS HERE ? NOT SO SURE
				if momentum[jj] is None:
					momentum[jj] = grad
				else :
					momentum[jj] = momentum_rate*momentum[jj] + grad
				
				if momentum_bias[jj] is None:
					momentum_bias[jj] = grad_bias
				else :
					momentum_bias[jj] = momentum_bias[jj] + grad_bias

				layer.weights -= learn_rate*momentum[jj]
				layer.hidden_biases -= learn_rate*momentum_bias[jj]
				#layer.
			if best_rmsd is not None:	
				rsmd_grad = (rmsd - best_rmsd) / best_rmsd
				print "Epoch %4d/%4d, RMS deviation = %7.5f, RMSD grad = %7.5f, early stopping is %d epoch ago" % (
					epoch + 1, iterations, rmsd, rsmd_grad, iter_since_best)
			else:
				print "Epoch %4d/%4d, RMS deviation = %7.5f" % (
					epoch + 1, iterations, rmsd)
			#,
			#if best_rmsd is not None: print (rmsd - best_rmsd) / best_rmsd
			#else: print
			rmsd_logger.append(rmsd)
			
			if early_stop:
				if best_rmsd is None or rsmd_grad < -2e-3 :
					if best_rmsd is None:
						best_weights = []
						for layer in self.layers :
							best_weights.append(layer.weights)
					else:
						for jj in range(self.mid * 2) :
							best_weights[jj] = self.layers[jj].weights
					best_rmsd = rmsd
					iter_since_best = 0
				else :
					iter_since_best += 1
					
				if iter_since_best >= max_epoch_without_improvement :
					for jj in range(self.mid * 2) :
						self.layers[jj].weights = best_weights[jj]
					
					print "Early stop -- best epoch %d / %d, RMS deviation = %7.4f" % (
											epoch + 1 - iter_since_best, iterations, best_rmsd)
					break
			
		rmsd_history = np.asarray(rmsd_logger)
		if iter_since_best > 0 :
			self.rmsd_history = rmsd_history[:-1*iter_since_best]
		else :
			self.rmsd_history = rmsd_history
					
		self.is_trained = True

	def plot_rmsd_history(self):
		import pylab as plt
				
		plt.figure()
		if self.is_pretrained:
			for layer in self.rbms :
				plt.plot(np.arange(np.size(layer.rmsd_history)), layer.rmsd_history, lw=2, 
						label="%d nodes" % layer.hidden_nodes)
		plt.semilogy(np.arange(np.size(self.rmsd_history)), self.rmsd_history, lw=2, label="Backprop")	
		plt.grid()
		plt.legend(loc="best")
		plt.xlabel("Epoch")
		plt.ylabel("RMS error")
		#plt.show()
		
	def save(self, fname):
		utils.writepickle(self, fname)
