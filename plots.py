import copy
import pylab as plt
import numpy as np
import os
import figures

def _nb_bins(data, norm=100, minb=10):
	return max(np.size(data) / norm, minb)

def vspans(d, c):
	plt.axvline(np.mean(d), color=c, lw=2)
	plt.axvspan(np.mean(d)-np.std(d), np.mean(d)+np.std(d), facecolor=c, alpha=0.1, edgecolor=c)
	
def hspans(d, c):
	plt.axhline(np.mean(d), color=c, lw=2)
	plt.axhspan(np.mean(d)-np.std(d), np.mean(d)+np.std(d), facecolor=c, alpha=0.1, edgecolor=c)

def hist(train, test, pca=None, xlabel=None):

	
	b = _nb_bins(train, 100, 20)
	_, _, patches = plt.hist(train, bins=b, normed=True, color='gold', label="Train set", alpha=.5)
	c = plt.getp(patches[0], 'facecolor')
	vspans(train, c)

	
	b = _nb_bins(test, 100, 20)
	_, _, patches = plt.hist(test, bins=b, normed=True, color='g', label="Test set", alpha=.5)
	c = plt.getp(patches[0], 'facecolor')
	vspans(test, c)
	
	if not pca is None:
		b = _nb_bins(pca, 100, 20)
		_, _, patches = plt.hist(pca, bins=b, normed=True, color='b', label="PCA set", alpha=.5)
		c = plt.getp(patches[0], 'facecolor')
		vspans(pca, c)
	
	plt.legend(loc="best")
	plt.grid()
	
	plt.xlabel(xlabel)

def lc_dataset_size(datasize, train, test):
	plt.plot(datasize, train, label='train error')
	plt.plot(datasize, test, label='test error')
	
	plt.legend(loc="best")
	plt.grid()
	
		
def visualise(ae, layer):
	if layer > ae.mid or layer < 0: 
		raise ValueError("Wrong layer number")
	
	W = self.layers[layer].weights
	plt.figure()
	plt.imshow(W, interpolation="nearest")
	plt.show()
	'''
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
	'''
	
def display_network(ae, layer_number=0, outpath=None):
	opt_normalize = True
	opt_graycolor = True
	
	A = copy.copy(ae.layers[layer_number].weights)
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
	
	# Saves or displays fig
	if outpath is None:
		plt.show()
	else:
		figures.savefig(os.path.join(outpath, "net_{:05d}".format(len(ae.train_history))), fig, fancy=True)

def display_train_history(self):
	
	plt.figure()
	
	for jj in range(self.mid):
		plt.plot(self.layers[jj].train_history, label="Layer %d" % jj, lw=2)
	
	plt.plot(self.train_history, lw=2, label="Fine-tune")
	plt.legend(loc='best')
	
	plt.show()
