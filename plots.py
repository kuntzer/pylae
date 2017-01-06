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
	
def display_network(ae, layer_number=0, outpath=None):
	
	A = copy.copy(ae.layers[layer_number].weights)

	# Rescale
	A = A - np.average(A)

	# Compute rows & cols
	(row, col) = A.shape
	sz = int(np.ceil(np.sqrt(row)))
	buf = 1
	n = np.ceil(np.sqrt(col))
	m = np.ceil(col / n)
	image = -np.ones(shape=(int(buf + m * (sz + buf)), int(buf + n * (sz + buf))))
	
	if not A[:, 0].size == sz * sz:
		raise ValueError("The layer has not the right dimensions...")

	k = 0
	for i in range(int(m)):
		for j in range(int(n)):
			if k >= col:
				continue

			clim = np.max(np.abs(A[:, k]))

			image[buf + i * (sz + buf):buf + i * (sz + buf) + sz, buf + j * (sz + buf):buf + j * (sz + buf) + sz] = \
					A[:, k].reshape(sz, sz) / clim

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
