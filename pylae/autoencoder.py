import numpy as np
import utils
import copy
import os
import classae
import scipy.optimize

class AutoEncoder(classae.GenericAutoEncoder):
	
	def __init__(self, name='ae', layer_type="gd", directory='', verbose=False):
		self.name = name
		self.is_pretrained = False
		self.is_trained = False
		self.verbose = verbose
		self.layer_type = layer_type
		
		self.filepath = os.path.join(directory, name, layer_type)
		if not os.path.isdir(self.filepath):
			os.makedirs(self.filepath)
		self.directory = directory
		self.train_history = []
	
	def pre_train(self, data, architecture, layers_type, learn_rate={'SIGMOID':3.4e-3, 'LINEAR':3.4e-4},
			initialmomentum=0.53, finalmomentum=0.93, iterations=2000, mini_batch=100,
			regularisation=0.001, max_epoch_without_improvement=30, early_stop=True):

		"""
		
		"""
		if self.layer_type == "gd" :
			import RBM_gd as RBM
		elif self.layer_type == "cd1" :
			import RBM_cd1 as RBM
		else:
			raise RuntimeError("Layer/pre-training type %s unknown." % self.layer_type)
		
		
		assert len(architecture) + 1 == len(layers_type)
		
		# Archiving the configuration
		rbms_config = {'architecture':architecture, 'layers_type':layers_type, 'learn_rate':learn_rate,
				'initialmomentum':initialmomentum, 'finalmomentum':finalmomentum,
				'regularisation':regularisation, 'iterations':iterations, 
				'max_epoch_without_improvement':max_epoch_without_improvement,
				'early_stop':early_stop}
		self.rbms_config = rbms_config
		
		layers = []
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
	
			layers.append(layer)
		
		self.is_pretrained = True
		self.set_autoencoder(layers)
	
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
					
				if layer.hidden_type == "RELU" :
					delta = delta * utils.relu_prime(layer.output)

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
				if layer.hidden_type == "RELU" :
					prim = utils.relu_prime(layer.output)
					
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
					method='L-BFGS-B', verbose=True, return_info=False, cost_fct='L2'):

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
			regularisation, sparsity, beta, data, cost_fct=cost_fct)
		
		if iterations >= 0:
			options_ = {'maxiter': iterations, 'disp': verbose}
			result = scipy.optimize.minimize(J, theta, method=method, jac=True, options=options_)
			opt_theta = result.x
		else:
			c, _ = J(theta)
			return c
		
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
	
	def cost(self, theta, indices, weights_shape, biases_shape, lambda_, sparsity, beta,\
			 data, cost_fct, log_cost=True):

		if cost_fct == 'cross-entropy':
			if beta != 0 or sparsity != 0:
				beta = 0
				sparsity = 0
				print 'WARNING: Cross-entropy does not support sparsity'

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
		
		wgrad = []
		bgrad = []
		
		############################################################################################
		# Cost function
		if cost_fct == 'L2':
			
			# Back-propagation
			delta = -(data - h)
		
			# Compute the gradient:
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
					if beta == 0:
						sparsity_delta = 0
						sparsity_cost = 0
					else:
						sparsity_delta = np.tile(- rho / rho_hat + (1 - rho) / (1 - rho_hat), (m, 1)).transpose()
						sparsity_cost += beta * np.sum(utils.KL_divergence(rho, rho_hat))
	
					delta = self.layers[jj+1].weights.dot(delta) + beta * sparsity_delta
					
				if self.layers[jj].hidden_type == 'SIGMOID':
					delta *= utils.sigmoid_prime(self.layers[jj].activation.T)
				if self.layers[jj].hidden_type == 'RELU':
					delta *= utils.relu_prime(self.layers[jj].activation.T)
				elif self.layers[jj].hidden_type == 'LINEAR':
					pass # Nothing more to do
				else:
					raise NotImplemented("Hidden type %s not implemented" % self.layers[jj].hidden_type)
				
				grad_w = delta.dot(self.layers[jj].input) / m + lambda_ * self.layers[jj].weights.T / m
				grad_b = np.mean(delta, axis=1)
				wgrad.append(grad_w.T)
				bgrad.append(grad_b)
					
			# Reverse the order since back-propagation goes backwards 
			wgrad = wgrad[::-1]
			bgrad = bgrad[::-1]
			
			# Computes the L2 norm + regularisation
			#TODO: COST MISSES THE COMPLETE SPARSITY !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
			cost = np.sum((h - data) ** 2) / (2 * m) + (lambda_ / 2) * \
				(sum([((self.layers[jj].weights)**2).sum() for jj in range(self.mid * 2)])) + \
				sparsity_cost
		elif cost_fct == 'cross-entropy':
			# Compute the gradients:
			# http://neuralnetworksanddeeplearning.com/chap3.html for details
			dEda = None
			
			for jj in range(self.mid * 2 - 1, -1, -1):
				#print jj, '-------' * 6
				# The output of the layer right before is
				if jj - 1 < 0:
					hn = data.T
				else:
					hn = self.layers[jj-1].output
				
				# If last layer, we compute the delta = output - expectation
				if dEda is None: 
					dEda = h - data
				else:
					wp1 = self.layers[jj+1].weights
					a = self.layers[jj].output
					dEda = wp1.dot(dEda) * (a * (1. - a)).T
					
				dEdb = np.mean(dEda, axis=1)

				dEdw = dEda.dot(hn) / m
				dEdw = dEdw.T
				
				wgrad.append(dEdw)
				bgrad.append(dEdb)
	
			# Reverse the order since back-propagation goes backwards 
			wgrad = wgrad[::-1]
			bgrad = bgrad[::-1]

			# Computes the cross-entropy
			cost = - np.sum(data * np.log(h) + (1. - data) * np.log(1. - h), axis=0) 
			cost = np.mean(cost)
		elif cost_fct == 'X_rmse_params': 
			raise NotImplemented()
		if log_cost:
			self.train_history.append(cost)
		
		#exit()
		# Returns the gradient as a vector.
		grad = self._roll(wgrad, bgrad, return_info=False)
		return cost, grad
	
	def get_cost(self, data, regularisation = 0., sparsity=0., beta=0.,
					method='L-BFGS-B', verbose=True, return_info=False, cost_fct='L2'):
		"""
		Returns the cost, but this is very slow!!!
		"""
		
		iterations = -1
		return self.fine_tune(data, iterations, regularisation, sparsity, beta,	method, verbose, return_info, cost_fct)
	
	def sgd(self, data, iterations, learning_rate, initial_momentum, final_momentum, minibatch=10, annealing=None, max_epoch_without_improvement=30, early_stop=True):
		"""
		Performes an Stochastic gradient descent (SGD) optimisation of the network.
		"""
		
		m = data.shape[1]
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
		v_mom = 0
		best_cost = 1e8
		
		batch_indices = np.arange(m)
		n_minibatches = np.int(np.ceil(m/minibatch))
		
		gamma = initial_momentum
		
		for epoch in range(iterations):
			np.random.shuffle(batch_indices)			
			for ibatch in range(n_minibatches+1):
				ids = batch_indices[ibatch*minibatch:(ibatch+1)*minibatch]
				batch = data[:,ids]
				
				_, thetan = self.cost(theta, indices, weights_shape, biases_shape,
				0, 0, 0, batch, cost_fct='cross-entropy', log_cost=False)
				v_mom = gamma * v_mom + learning_rate * thetan
				theta -= v_mom
			
			actual = self.feedforward(data.T)
			cost = utils.cross_entropy(data.T, actual)
			print 'Epoch %4d/%4d:\t%e' % (epoch+1, iterations, cost)

			self.train_history.append(cost)
			
			if cost <= best_cost :
				best_cost = cost
				iter_best = epoch
				
			if epoch - iter_best > max_epoch_without_improvement :
				print 'STOP: %d epoches without improvment' % max_epoch_without_improvement
				break
			
			
			if annealing is not None:
				learning_rate /= (1. + float(epoch) / annealing)
				
			if epoch > 100:
				gamma = final_momentum
			else:
				gamma = initial_momentum + (final_momentum - initial_momentum) * utils.sigmoid(epoch - 50)
			print learning_rate, gamma
			
		###########################################################################################
		# Unroll the state vector and saves it to self.	
		for jj in range(self.mid * 2):
			w, b = self._unroll(theta, jj, indices, weights_shape, biases_shape)
			
			self.layers[jj].weights = w
			self.layers[jj].hidden_biases = b
			
		# We're done !
		self.is_trained = True

	def asgd(self, data, iterations, learning_rate=0.001, minibatch=10, decay_rate=0.9, max_epoch_without_improvement=30, early_stop=True):
		"""
		http://sebastianruder.com/optimizing-gradient-descent/index.html#adadelta
		"""
		
		m = data.shape[1]
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
		best_cost = 1e8
		iter_best = 0
		
		eps = 1e-12 
		Dt2 = np.zeros_like(theta)
		Eg2 = np.zeros_like(theta)
		
		
		for epoch in range(iterations):
			cost, thetan = self.cost(theta, indices, weights_shape, biases_shape,
				0, 0, 0, data, cost_fct='cross-entropy', log_cost=False)
			
			Eg2 = decay_rate * Eg2 + (1. - decay_rate) * thetan * thetan
			
			delta_theta = - learning_rate / np.sqrt(Eg2 + eps) * thetan
			
			theta = theta + delta_theta#- np.sqrt(Dt2 + eps) / np.sqrt(Eg2 + eps) * thetan
			
			print 'Epoch %4d/%4d:\t%e' % (epoch+1, iterations, cost)

			self.train_history.append(cost)
			
			if cost <= best_cost :
				best_cost = cost
				iter_best = epoch
				
			if epoch - iter_best > max_epoch_without_improvement :
				print 'STOP: %d epoches without improvment' % max_epoch_without_improvement
				break
						
		###########################################################################################
		# Unroll the state vector and saves it to self.	
		for jj in range(self.mid * 2):
			w, b = self._unroll(theta, jj, indices, weights_shape, biases_shape)
			
			self.layers[jj].weights = w
			self.layers[jj].hidden_biases = b
			
		# We're done !
		self.is_trained = True

