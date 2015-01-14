import pylab as plt
import numpy as np

def _nb_bins(data, norm=100, minb=10):
	return max(np.size(data) / norm, minb)

def hist(train, test, validation=None):
	b = _nb_bins(train, 100, 10)
	plt.hist(train, normed=True, label="Train set", alpha=.5, bins=b)
	
	b = _nb_bins(test, 100, 10)
	plt.hist(test, normed=True, label="Test set", alpha=.5)
	
	if not validation is None:
		b = _nb_bins(validation, 100, 10)
		plt.hist(validation, normed=True, label="Test set", alpha=.5)
	
	plt.legend(loc="best")
	plt.grid()