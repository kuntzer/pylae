import numpy as np
import utils
import copy
import os

class AutoEncoder():
	
	def __init__(self, name='ae', rbm_type="gd", directory='', verbose=False):
		self.name = name
		self.is_pretrained = False
		self.is_trained = False
		self.verbose = verbose
		self.rbm_type = rbm_type
		
		self.filepath = os.path.join(directory, name, rbm_type)
		if not os.path.isdir(self.filepath):
			os.makedirs(self.filepath)
		self.directory = directory
		
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
		
	def save_rbms(self, filepath=None):
		"""
		Saves the pre-trained rbms to a pickle file.
		
		:param filepath: the path of the file, default path is defined in the init / rmbs.pkl
		"""
		if filepath is None:
			filepath = self.filepath
		
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
			print "encoding, for dimensions", layer.weights.shape
			data = layer.feedforward(data)
			
		return data
		
	def decode(self, data):
		"""
		Decodes the data using the decoder
		:param data: encoded data
		
		:returns: decoded data
		"""
		for layer in self.layers[self.mid:] :
			print "decoding, for dimensions", layer.weights.shape
			data = layer.feedforward(data)
			
		return data
	
	def backpropagation(self, data, iterations=500, learn_rate=0.13, momentum_rate=0.83, 
					max_epoch_without_improvement=30, regularisation = 0.0001, early_stop=True):
		
		if not self.is_pretrained: 
			raise RuntimeError("The autoencoder is not pre-trained.")

		
		N = np.shape(data)[0]
		momentum = [None] * len(self.layers)
		momentum_bias = [None] * len(self.layers)
		
		assert N > 0
		rmsd_logger = []

		best_rmsd = None
		iter_since_best = 0
		
		momentum_rate_save = momentum_rate
		
		for epoch in range(iterations) :
			a_l = self.feedforward(data)
			# Start backpropagation
			if epoch > 15:
				momentum_rate = momentum_rate_save
			else:
				momentum_rate = 0.5
			
			# Initialise the accumulated derivating using square error E = 0.5*(T-Y)^2
			diff_y_an = data - a_l
			delta =  -diff_y_an # dE/dz
			rmsd = np.sqrt(np.mean(delta*delta)) / 2.
			
			# Backpropagate through the top to bottom
			for jj in range(self.mid * 2 - 1, -1, -1):
				layer = self.layers[jj]

				if layer.hidden_type == "SIGMOID" :
					sigmoid_deriv = (1.0 - layer.output) * layer.output
					delta = delta * sigmoid_deriv

					grad_bias = np.mean(delta, axis=0)# dJ/dB is error only, no regularisation for biases
					
				elif layer.hidden_type == "LINEAR":
					grad_bias = np.mean(layer.output - layer.hidden_biases, axis=0)/N
					
					delta = delta
				else :
					raise ValueError("Type of layer not recognised")

				grad = np.dot(layer.input.T, delta)/N + regularisation * layer.weights # dz/dW partial derivative for weights

				if np.any(grad) > 1e9 or np.any(grad) is np.nan :
					raise ValueError("Weights have blown to infinite! \
						Try reducing the learning rate.")
					
				# no point accumulating gradients for the first layer
				if jj > 0:
					delta = np.dot(delta, layer.weights.T) 

				if momentum[jj] is None:
					momentum[jj] = grad
				else :
					momentum[jj] = momentum_rate*momentum[jj] + grad
				
				if momentum_bias[jj] is None:
					momentum_bias[jj] = grad_bias
				else :
					momentum_bias[jj] = momentum_rate*momentum_bias[jj] + grad_bias

				layer.weights -= learn_rate*momentum[jj]
				layer.hidden_biases -= learn_rate*momentum_bias[jj]

			if best_rmsd is not None:	
				rsmd_grad = (rmsd - best_rmsd) / best_rmsd
				print "Epoch %4d/%4d, RMS deviation = %7.5f, RMSD grad = %7.5f, early stopping is %d epoch ago" % (
					epoch + 1, iterations, rmsd, rsmd_grad, iter_since_best)
			else:
				print "Epoch %4d/%4d, RMS deviation = %7.5f" % (
					epoch + 1, iterations, rmsd)

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


	def backpropagation_with_sparsity(self, data, iterations=500, learn_rate=0.13, momentum_rate=0.83, 
					max_epoch_without_improvement=30, regularisation = 0., sparsity=None, beta=3.,
					early_stop=True):
		
		if not self.is_pretrained: 
			raise RuntimeError("The autoencoder is not pre-trained.")

		
		N = np.shape(data)[0]
		momentum = [None] * len(self.layers)
		momentum_bias = [None] * len(self.layers)
		
		assert N > 0
		rmsd_logger = []

		best_rmsd = None
		iter_since_best = 0
		
		momentum_rate_save = momentum_rate
		
			
		#TODO: WHAT THE F*** IS GOING ON WITH THE SPARSITY ?
		# LET'S ISSUE A WARNING AND TURN OFF SPARSITY:
		if sparsity is not None:
			print '!!!!!!!! There is an unknown issue with the sparsity constraint'
			print 'if you want to use the sparisity use the autoencoder.fine_tune() method'
			print 'THIS AN EXPERIMENTAL METHOD'
			sparsity = None
		
		for epoch in range(iterations) :
			a_l = self.feedforward(data)
			# Start backpropagation
			if epoch > 15:
				momentum_rate = momentum_rate_save
			else:
				momentum_rate = 0.5
			# Initialise the accumulated derivating using square error E = 0.5*(T-Y)^2
			diff_y_an = data - a_l
			delta =  -diff_y_an # dE/dz
			rmsd = np.sqrt(np.mean(delta*delta)) / 2.
			
			# Backpropagate through the top to bottom
			for jj in range(self.mid * 2 - 1, -1, -1):
				layer = self.layers[jj]
				rho_hat = np.mean(layer.output, axis=0)
				#print np.shape(rho_hat), 
				#print np.shape(delta), '<<<'

				#print np.shape(layer.weights)
				#print np.shape(layer.hidden_biases)
				
				#print np.shape(KL)
				#print np.shape((1.0 - layer.output) * layer.output)
				#print jj, 
				if layer.hidden_type == "SIGMOID" :
					# later part is error * gradient of cost function (derivative of sigmoid s(x) = s(x){1-s(x)}
					sigmoid_deriv = (1.0 - layer.output) * layer.output
					#print sparsity, np.mean(rho_hat)
					prim = sigmoid_deriv
					
					grad_bias = np.mean(delta, axis=0)# dJ/dB is error only, no regularisation for biases
					
				elif layer.hidden_type == "LINEAR":
					#grad_bias = np.mean(delta, axis=0)
					grad_bias = np.mean(layer.output - layer.hidden_biases, axis=0)/N
					
					
					prim = 1
					
					#prim = (1.0 - layer._sigmoid(layer.output)) * layer._sigmoid(layer.output)
				else :
					raise ValueError("Type of layer not recognised")
				
				#print jj, np.shape(grad_bias); #exit()
				rho = np.tile(sparsity, np.shape(layer.output)[1])
				if sparsity is not None and jj < self.mid * 2 - 1:
					#KL = - sparsity / rho_hat #+ (1. - sparsity) / (1. - rho_hat)
					#print np.shape(layer.input)
					KL = np.tile(- rho / rho_hat + (1 - rho) / (1 - rho_hat), (N, 1))#.transpose()
					#print np.shape(layer.output), N
					#print np.shape(KL), np.shape(sigmoid_deriv)
					#exit()
					spar = beta * KL
					#spar = np.dot(layer.input.T, spar)
				else:
					spar = 0.
				#print

				print jj, np.shape(spar)
				delta = (delta + spar) * prim

				#print KL
				#print np.shape(KL),
				#print np.shape(regularisation * layer.weights)
				#print np.shape(spar),
				#print np.shape(delta)
				#print np.shape(layer.input.T),
				#print jj, np.mean(spar)
				#print np.shape(spar)
				grad = np.dot(layer.input.T, delta)/N 
				grad += regularisation * layer.weights # dz/dW partial derivative for weights
				#print np.shape(grad)
				#exit()
				#print spar
				if np.any(grad) > 1e9 or np.any(grad) is np.nan :
					raise ValueError("Weights have blown to infinite! \
						Try reducing the learning rate.")
					
				# no point accumulating gradients for the first layer
				if jj > 0:
					delta = np.dot(delta, layer.weights.T) 

				if momentum[jj] is None:
					momentum[jj] = grad
				else :
					momentum[jj] = momentum_rate*momentum[jj] + grad
				
				if momentum_bias[jj] is None:
					momentum_bias[jj] = grad_bias
				else :
					momentum_bias[jj] = momentum_rate*momentum_bias[jj] + grad_bias

				layer.weights -= learn_rate*momentum[jj]
				layer.hidden_biases -= learn_rate*momentum_bias[jj]
			#exit()	
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
		
	def fine_tune(self, data, iterations=400, regularisation = 0.003, sparsity=0.1, beta=3.,
					method='L-BFGS-B', verbose=True, return_info=False):
		
		import scipy.optimize

		# TODO:
		# we could change this in the whole code so that we don't need to transpose too many
		# times. However, transpose is an a fast operation as it returns a view of the array...
		data = data.T
		###########################################################################################
		# Initialisation of the weights and bias
		
		# Copy the weights and biases into a state vector theta
		weights = []
		biases = []
		for jj in range(self.mid * 2):
			weights.append(copy.copy(self.layers[jj].weights))
			biases.append(self.layers[jj].hidden_biases) 
			
		theta, indices, weights_shape, biases_shape = self._roll(weights, biases)
		del weights, biases

		###########################################################################################
		# Optimisation of the weights and biases according to the cost function
		# under the constraint of regularisation and sparsity given by the user 
		J = lambda x: self.cost(x, indices, weights_shape, biases_shape,
			regularisation, sparsity, beta, data)
		
		options_ = {'maxiter': iterations, 'disp': verbose}
		result = scipy.optimize.minimize(J, theta, method=method, jac=True, options=options_)
		opt_theta = result.x
		
		#utils.writepickle(opt_theta, 'opt_theta.pkl')
		#opt_theta = utils.readpickle('opt_theta.pkl')
		
		###########################################################################################
		# Unroll the state vector and saves it to self.	
		for jj in range(self.mid * 2):
			w, b = self._unroll(opt_theta, jj, indices, weights_shape, biases_shape)
			
			self.layers[jj].weights = w
			self.layers[jj].hidden_biases = b
		
		if verbose: print result
		
		# We're done !
		self.is_trained = True
		
		# If the user requests the informations about the minimisation...
		if return_info: return result
	
	def cost(self, theta, indices, weights_shape, biases_shape, lambda_, sparsity, beta, data):

		# Unrolling the weights and biases
		for jj in range(self.mid * 2):
			w, b = self._unroll(theta, jj, indices, weights_shape, biases_shape)

			self.layers[jj].weights = w
			self.layers[jj].hidden_biases = b
	
		# Number of training examples
		m = data.shape[1]

		# Forward pass
		h = self.feedforward(data.T).T

		# Sparsity
		sparsity_cost = 0
		
		# Back-propagation
		delta = -(data - h)
		wgrad = []
		bgrad = []
		
		
		for jj in range(self.mid * 2 - 1, -1, -1):

			if jj < self.mid * 2 - 1:
				# TODO: Sparsity: do we want it at every (hidden) layer ?? 
				"""print jj
				print np.shape(self.layers[2].output.T)
				
				print hn
				print self.layers[2].hidden_nodes
				print m 
				exit()"""
				hn = self.layers[jj].output.T.shape[0]
				#print hn, np.shape(self.layers[2].output.T)
				rho_hat = np.mean(self.layers[jj].output.T, axis=1)
				rho = np.tile(sparsity, hn)
				#print np.shape(rho_hat), rho.shape
				sparsity_delta = np.tile(- rho / rho_hat + (1 - rho) / (1 - rho_hat), (m, 1)).transpose()
				sparsity_cost += beta * np.sum(utils.KL_divergence(rho, rho_hat))

				delta = self.layers[jj+1].weights.dot(delta) + beta * sparsity_delta
				
			delta *= utils.sigmoid_prime(self.layers[jj].activation.T)
			
			grad_w = delta.dot(self.layers[jj].input) / m + lambda_ * self.layers[jj].weights.T
			grad_b = np.mean(delta, axis=1)
			wgrad.append(grad_w.T)
			bgrad.append(grad_b)
				
		# Reverse the order since back-propagation goes backwards 
		wgrad = wgrad[::-1]
		bgrad = bgrad[::-1]
		
		# Cost function
		# COST MISSES THE COMPLETE SPARSITY !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
		# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
		# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
		# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
		# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
		cost = np.sum((h - data) ** 2) / (2 * m) + (lambda_ / 2) * \
			(sum([((self.layers[jj].weights)**2).sum() for jj in range(self.mid * 2)])) + \
			sparsity_cost

		# Returns the gradient as a vector.
		grad = self._roll(wgrad, bgrad, return_info=False)
		return cost, grad
		
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

	def plot_rmsd_history(self, save=False, filepath=None):
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
		
		if save:
			if filepath == None: filepath = os.path.join(self.filepath, 'rmsd_history.png')
			plt.savefig(filepath)
		
	def save(self, filepath=None):
		if filepath is None:
			filepath = self.filepath
			
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

	def display_network(self, layer_number=0):
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
	
		image = np.ones(shape=(buf + m * (sz + buf), buf + n * (sz + buf)))
	
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
	
		plt.figure()
		plt.imshow(image, interpolation="nearest", cmap=plt.get_cmap('gray'))
	
		#plt.show()
	