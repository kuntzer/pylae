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
				
				rho_hat = np.mean(delta, axis=0)
				#print np.shape(rho_hat), 
				#print np.shape(delta), '<<<'

				layer = self.layers[jj]
				
				#print np.shape(KL)
				#print np.shape((1.0 - layer.output) * layer.output)
				#print jj, 
				if layer.hidden_type == "SIGMOID" :
					# later part is error * gradient of cost function (derivative of sigmoid s(x) = s(x){1-s(x)}
					sigmoid_deriv = (1.0 - layer.output) * layer.output
					#print sparsity, np.mean(rho_hat)
					delta = delta * sigmoid_deriv
					if sparsity is not None and jj == self.mid * 2 - 1 and jj < 0:
						KL = - sparsity / rho_hat# + (1. - sparsity) / (1. - rho_hat)
						#print KL
						#exit()
						spar = beta * KL * (1.0 - layer.output) * layer.output
						spar = np.dot(layer.input.T, spar)
					else:
						spar = 0.

					
					grad_bias = np.mean(delta, axis=0)# dJ/dB is error only, no regularisation for biases
					
				elif layer.hidden_type == "LINEAR":
					grad_bias = np.mean(layer.output - layer.hidden_biases, axis=0)/N
					spar = 0.
					
					delta = delta
				else :
					raise ValueError("Type of layer not recognised")

				#print KL
				#print np.shape(KL),
				#print np.shape(regularisation * layer.weights)
				#print np.shape(spar),
				#print np.shape(delta)
				#print np.shape(layer.input.T),
				
				#print np.shape(spar)
				grad = np.dot(layer.input.T, delta)/N + regularisation * layer.weights + spar # dz/dW partial derivative for weights
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
		#plt.show()
		
	def save(self, filepath=None):
		utils.writepickle(self, os.path.join(self.filepath, 'ae.pkl'))

	def load(self, filepath=None):
		
		if filepath is None:
			filepath = self.filepath
		
		self.load_rbms(filepath)
		
		self = utils.readpickle(os.path.join(self.filepath, 'ae.pkl'))
		self.is_trained = True
		
	def visualise(self, layer):
		"""
		void DisplayWeights(const mat &weights)
{
    // Scale so that negative values range from [0, 128] and positive from [128, 255]

    int width = sqrt(weights.n_cols); // assume square image

    for(unsigned int j=0; j < weights.n_rows; j++) {
        double a = min(weights.row(j));
        double b = max(weights.row(j));
        double s = max(fabs(a), fabs(b));

        mat tmp = (weights.row(j) / s)*127 + 128;

        cv::Mat img(width, width, CV_8U);

        for(int k=0; k < width*width; k++) {
            int v = tmp(k);

            if(v < 0) v = 0;
            if(v >= 256) v = 255;

            img.at<uchar>(k) = v;
        }

        cv::resize(img, img, cv::Size(width*10, width*10), 0, 0, cv::INTER_NEAREST);
        cv::imshow("main", img);
        cv::waitKey(0);
    }
}
		"""
		import pylab as plt
		print self.mid
		#if layer >= self.mid or layer < 1: 
		#	raise ValueError("Wrong layer number")
			
		
		W = self.layers[layer].weights
		print W
		nin, nout = np.shape(W)
		
		snout = int(np.sqrt(nout))
		f, axes = plt.subplots(snout, snout)#, sharex='col', sharey='row')
		
		x = 0
		y = 0
		for ii in range(nout):
			
			img = W[:,ii] / np.sqrt(np.sum(W[:,ii]**2))
			
			axes[x, y].imshow(img.reshape([np.sqrt(nin),np.sqrt(nin)]), interpolation="nearest", cmap=plt.get_cmap('gray'))
			
			x += 1
			if x >= np.sqrt(nout):
				x = 0
				y += 1
			 
			"""print ii
			plt.figure()
			print np.shape(img)
			print np.sqrt(nout)
			plt"""
		plt.setp([[a.get_xticklabels() for a in b] for b in axes[:,]], visible=False)
		plt.setp([[a.get_yticklabels() for a in b] for b in axes[:,]], visible=False)
		
		
