import numpy as np
import utils
import layer

class RBM(layer.AE_layer):
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
		self.mini_batch = mini_batch
		self.iterations = iterations
		self.max_epoch_without_improvement = max_epoch_without_improvement
		self.early_stop = early_stop
	
	def train(self, data_train, initialmomentum, finalmomentum, learn_rate_w,
			learn_rate_visb, learn_rate_hidb, weightcost):

		Nd, numdims = np.shape(data_train)
		N = self.mini_batch
		n_tot_batches = np.int(np.ceil(Nd/self.mini_batch))
		
		vishid = 0.1 * np.random.randn(numdims, self.hidden_nodes)
		
		hidbiases = np.zeros(self.hidden_nodes)
		visbiases = np.zeros(numdims)
		
		vishidinc = np.zeros_like(vishid)
		hidbiasinc = np.zeros_like(hidbiases)
		visbiasinc = np.zeros_like(visbiases)
		"""
		pos_hidprobs = np.zeros([N, self.hidden_nodes])
		neg_hidprobs = np.zeros_like(pos_hidprobs)
		pos_prods = np.zeros_like(vishid)
		neg_prods = np.zeros_like(vishid)

		"""
		batchposidprobs = np.empty([N, self.hidden_nodes, n_tot_batches])
		
		rmsd_logger = []
		best_weights = np.zeros_like(vishid)
		best_hidden_biases = np.zeros_like(hidbiases)
		best_rmsd = None
		iter_since_best = 0
		
		for epoch in range(self.iterations):
			errsum = 0
			rmsd = 0
			##print "Epoch %d / %d" % (epoch + 1, self.iterations)
			for batch in range(n_tot_batches):
				#print "  epoch %d / %d -- batch %d / %d" % (epoch + 1, self.iterations,\
				#										 batch + 1, n_tot_batches)
				
				start = (batch % n_tot_batches) * self.mini_batch
				end = start + self.mini_batch			
				if end >= Nd : end = Nd
				data = data_train[start:end]

				## START POSITIVE PHASE ##################################################
				nw = np.dot(data, vishid) + np.tile(hidbiases, (N, 1))
				if self.hidden_type == "SIGMOID":
					pos_hidprobs = utils.sigmoid(nw)
				elif self.hidden_type == "LINEAR": 
					pos_hidprobs = nw
				
				if epoch >= self.iterations - 1:
					batchposidprobs[:,:,batch] = pos_hidprobs			
				pos_prods = np.dot(data.T, pos_hidprobs)
				pos_hidact = np.sum(pos_hidprobs, 0)
				pos_visact = np.sum(data, 0)
				## END OF POSITIVE PHASE ################################################
				if self.hidden_type == "SIGMOID":
					ran = np.random.rand(N, self.hidden_nodes)
					pos_hidstates = pos_hidprobs > ran
				elif self.hidden_type == "LINEAR":
					ran = np.random.randn(N, self.hidden_nodes)
					pos_hidstates = pos_hidprobs + ran
					
				## START NEGATIVE PHASE #################################################
				nw = np.dot(pos_hidstates,vishid.T) + np.tile(visbiases, (N,1))
				
				# TODO: Do this only if visible type is sigmoid see C++ line 262 and next
				if self.visible_type == "SIGMOID":
					neg_data = utils.sigmoid(nw)
				else:
					neg_data = nw
				
				nw = np.dot(neg_data, vishid) + np.tile(hidbiases, (N, 1))

				if self.hidden_type == "SIGMOID":
					neg_hidprobs = utils.sigmoid(nw)
				else: 
					neg_hidprobs = nw
				
				neg_prods = np.dot(neg_data.T, neg_hidprobs)
				neg_hidact = np.sum(neg_hidprobs, 0)
				neg_visact = np.sum(neg_data, 0)

				## END OF NEGATIVE PHASE ################################################
				
				errsum += np.sum((data-neg_data)*(data-neg_data))
				rmsd += np.sqrt(np.mean((data-neg_data)*(data-neg_data)))
				#print rmsd; exit()
				
				if epoch > 5:
					momentum = finalmomentum
				else:
					momentum = initialmomentum 
					
				## UPDATE WEIGHTS AND BIASES ############################################
				vishidinc = momentum * vishidinc + learn_rate_w * \
					((pos_prods - neg_prods)/N - weightcost * vishid)
				visbiasinc = momentum * visbiasinc + learn_rate_visb/N * (pos_visact - neg_visact)
				hidbiasinc = momentum * hidbiasinc + learn_rate_hidb/N * (pos_hidact - neg_hidact)
				
				vishid += vishidinc
				visbiases += visbiasinc
				hidbiases += hidbiasinc
				
			print "Epoch %4d/%4d, RMS deviation = %7.4f" % (epoch + 1, self.iterations, rmsd)
			rmsd_logger.append(rmsd)
			
			if self.early_stop:			
				if best_rmsd is None or (rmsd - best_rmsd) / best_rmsd < -1e-3 :
					best_weights = vishid
					best_hidden_biases = hidbiases
					best_visible_biases = visbiases
					best_rmsd = rmsd
					iter_since_best = 0
				else :
					iter_since_best += 1
					
				if iter_since_best >= self.max_epoch_without_improvement :
					print "Early stop -- best epoch %d / %d, RMS deviation = %7.4f" % (
											epoch + 1 - iter_since_best, self.iterations, best_rmsd)
					break
			else: 
				best_weights = vishid
				best_hidden_biases = hidbiases
				best_visible_biases = visbiases
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
