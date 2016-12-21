import numpy as np
import utils
import copy
import os

class GenericAutoEncoder():
	
	def __init__(self, name='ae', rbm_type="gd", directory='', verbose=False, mkdir=True):
		self.name = name
		self.is_pretrained = False
		self.is_trained = False
		self.verbose = verbose
		self.rbm_type = rbm_type
		
		self.filepath = os.path.join(directory, name, rbm_type)
		if not os.path.isdir(self.filepath) and mkdir:
			os.makedirs(self.filepath)
		self.directory = directory
		self.train_history = []
		
	def info(self):
		"""
		Prints out all the variables for a given galaxy of the class
		"""
		import inspect
	
		message = "All variables available for this autoencoder"		
		print message
		print '-'*len(message)
		attributes = inspect.getmembers(self, lambda a:not(inspect.isroutine(a)))
		for a in attributes:
			if (a[0].startswith('__') and a[0].endswith('__')): continue
			print a[0], "=", a[1]
			
	def clear_mem(self):
		self.rbms = None
			
		for jj in range(len(self.layers)):
			self.layers[jj].output = None
			self.layers[jj].activation = None
			self.layers[jj].input = None
		
	def set_autoencoder(self, encoder):
		"""
		From a list of layers object `encoder`, create an autoencoder by adding in a reversed order 
		of the encoder list to the encoder. Swaps the weights and biases too.
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
			
		
	def feedforward(self, data, dropout=None):
		"""
		Encodes and decodes the data using the full autoencoder
		:param data: The data in the same format than the training data
		
		:returns: the fed-forward data
		"""
		for layer in self.layers :
			#print layer.hidden_nodes, '-'*30
			data = layer.feedforward_memory(data, dropout=dropout)

		return data
	
	def feedforward_to_layer(self, data, j, dropout=None):
		"""
		Encodes and decodes the data using the full autoencoder
		:param data: The data in the same format than the training data
		:param j: the layer to stop at 
		
		:returns: the fed-forward data
		"""
		for layer in self.layers[:j+1] : # TODO: +1, really?
			data = layer.feedforward(data, dropout=dropout)

		return data
	
	def pre_train(self, data, architecture, layers_type, learn_rate={'SIGMOID':3.4e-3, 'LINEAR':3.4e-4},
			initialmomentum=0.53, finalmomentum=0.93, iterations=2000, mini_batch=100,
			regularisation=0.001, max_epoch_without_improvement=30, early_stop=True):

		"""
		
		"""
		if self.rbm_type == "gd" :
			import RBM_gd as RBM
		elif self.rbm_type == "cd1" :
			import RBM_cd1 as RBM
		else:
			raise RuntimeError("RBM_gd training type %s unknown." % self.rbm_type)
		
		
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
					finalmomentum=finalmomentum, weightcost=regularisation)
			data = layer.feedforward(data)
	
			rbms.append(layer)
		
		self.is_pretrained = True
		self.set_autoencoder(rbms)
		
	def save_rbms(self, filepath=None, clear_mem=True):
		"""
		Saves the pre-trained rbms to a pickle file.
		
		:param filepath: the path of the file, default path is defined in the init / rmbs.pkl
		"""
		if filepath is None:
			filepath = self.filepath
			
		if clear_mem:
			self.clear_mem()
		
		utils.writepickle(self.rbms, os.path.join(filepath, 'rmbs.pkl'))
		
	def load_rbms(self, filepath=None):
		"""
		Reads the pre-trained rbms from a pickle file.
		
		:param filepath: the path of the file, default path is defined in the init / rmbs.pkl
		"""
		
		if filepath is None:
			filepath = self.filepath
		
		rbms = utils.readpickle(os.path.join(filepath, 'rmbs.pkl'))
		self.set_autencoder(rbms)
		self.is_pretrained = True
		
	def encode(self, data):
		"""
		Encodes the data using the encoder
		:param data: The data in the same format than the training data
		
		:returns: encoded data
		"""
		for layer in self.layers[:self.mid] :
			if self.verbose: print "encoding, for dimensions", layer.weights.shape
			data = layer.feedforward(data)
			
		return data
		
	def decode(self, data):
		"""
		Decodes the data using the decoder
		:param data: encoded data
		
		:returns: decoded data
		"""
		for layer in self.layers[self.mid:] :
			if self.verbose: print "decoding, for dimensions", layer.weights.shape
			data = layer.feedforward(data)
			
		return data

	def _roll(self, weightslist, biaseslist, return_info=True):
		biases_shape = []
		biases = []
		weights_shape = []
		weights = []
		indices = np.array([0])
		
		# Copy the weights and biases into a state vector theta
		for jj in range(self.mid * 2):
			w = weightslist[jj].T
			b = biaseslist[jj]
			biases = np.concatenate((biases, b))
			biases_shape.append(np.shape(b)[0])
			
			weights_shape.append(np.shape(w))
			reshaped = weights_shape[-1][0] * weights_shape[-1][1]
			w = w.reshape(reshaped)

			indices = np.concatenate((indices, [indices[-1]+reshaped]))
			weights = np.concatenate((weights, w))
			
		theta = np.concatenate((weights, biases))
		
		# Let's remember the indices for simplicity
		for ib in biases_shape:
			indices = np.concatenate((indices, [indices[-1]+ib]))
		
		if return_info:
			return theta, indices, weights_shape, biases_shape
		else:
			return theta
		
	def _unroll(self, theta, layer, indices, weights_shape, biases_shape):
		w = theta[indices[layer]: indices[layer+1]].reshape(weights_shape[layer][0], weights_shape[layer][1]).transpose() 
		b = theta[indices[layer + 2. * self.mid]: indices[layer + 1 + 2. * self.mid]].reshape(biases_shape[layer])

		return w, b
	
	def save(self, filepath=None, clear_mem=True):
		if filepath is None:
			filepath = self.filepath
			
		if clear_mem:
			self.clear_mem()
			
		utils.writepickle(self, os.path.join(filepath, 'ae.pkl'))

	def load(self, filepath=None):
		
		if filepath is None:
			filepath = self.filepath
		
		self.load_rbms(filepath)
		
		self = utils.readpickle(os.path.join(filepath, 'ae.pkl'))
		self.is_trained = True
		
	def visualise(self, layer):
		import pylab as plt
		if layer > self.mid or layer < 0: 
			raise ValueError("Wrong layer number")
		
		W = self.layers[layer].weights
		plt.figure()
		plt.imshow(W, interpolation="nearest")
		plt.show()
		exit()
		nin, nout = np.shape(W)
		
		snout = int(np.sqrt(nout))
		
		
		x = 0
		y = 0
		for ii in range(nout):
			"""print np.shape(np.sqrt(np.sum(W[ii]**2)))
			print np.shape(np.sqrt(np.sum(W[:,ii]**2)))
			
			exit()"""
			#print np.shape(W[:,ii])
			#print np.shape(W[ii])

			arr =  W[:,ii].T
			img = arr / np.amax(arr) #/ np.sqrt(np.sum(W[ii]**2))
			img = img.reshape([1,64])
			plt.imshow(img)
			plt.show()
			exit()
			
			#print np.shape(img)
			#exit()
			f, axes = plt.subplots(snout, snout)#, sharex='col', sharey='row')
			axes[x, y].imshow(img.reshape([np.sqrt(nin),np.sqrt(nin)]), interpolation="nearest", cmap=plt.get_cmap('gray'))
			
			x += 1
			if x >= np.sqrt(nout):
				x = 0
				y += 1

		plt.setp([[a.get_xticklabels() for a in b] for b in axes[:,]], visible=False)
		plt.setp([[a.get_yticklabels() for a in b] for b in axes[:,]], visible=False)

	def display_network(self, layer_number=0, show=True):
		import pylab as plt
		opt_normalize = True
		opt_graycolor = True
		
		A = copy.copy(self.layers[layer_number].weights)
		#print np.shape(A)

		# Rescale
		A = A - np.average(A)
	
		# Compute rows & cols
		(row, col) = A.shape
		sz = int(np.ceil(np.sqrt(row)))
		buf = 1
		n = np.ceil(np.sqrt(col))
		m = np.ceil(col / n)
	
		image = np.ones(shape=(int(buf + m * (sz + buf)), int(buf + n * (sz + buf))))
	
		if not opt_graycolor:
			image *= 0.1
	
		k = 0
		for i in range(int(m)):
			for j in range(int(n)):
				if k >= col:
					continue
	
				clim = np.max(np.abs(A[:, k]))
	
				if opt_normalize:
					image[buf + i * (sz + buf):buf + i * (sz + buf) + sz, buf + j * (sz + buf):buf + j * (sz + buf) + sz] = \
						A[:, k].reshape(sz, sz) / clim
				else:
					image[buf + i * (sz + buf):buf + i * (sz + buf) + sz, buf + j * (sz + buf):buf + j * (sz + buf) + sz] = \
						A[:, k].reshape(sz, sz) / np.max(np.abs(A))
				k += 1
	
		fig = plt.figure()
		plt.imshow(image, interpolation="nearest", cmap=plt.get_cmap('gray'))
		
		#return fig
		if show: plt.show()
	
	def display_train_history(self):
		import pylab as plt
		
		plt.figure()
		
		for jj in range(self.mid):
			plt.plot(self.layers[jj].train_history, label="Layer %d" % jj, lw=2)
		
		plt.plot(self.train_history, lw=2, label="Fine-tune")
		plt.legend(loc='best')
		
		plt.show()