import pylab as plt
import numpy as np

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