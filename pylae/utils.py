import cPickle as pickle 
import gzip
import os
import numpy as np

def readpickle(filepath):
	"""
    I read a pickle file and return whatever object it contains.
    If the filepath ends with .gz, I'll unzip the pickle file.
    """
	pkl_file = open(filepath, 'rb')
	obj = pickle.load(pkl_file)
	pkl_file.close()
	return obj

def writepickle(obj, filepath, protocol = -1):
	"""
	I write your python object obj into a pickle file at filepath.
	If filepath ends with .gz, I'll use gzip to compress the pickle.
	Leave protocol = -1 : I'll use the latest binary protocol of pickle.
	"""
	if os.path.splitext(filepath)[1] == ".gz":
		pkl_file = gzip.open(filepath, 'wb')
	else:
		pkl_file = open(filepath, 'wb')

	pickle.dump(obj, pkl_file, protocol)
	pkl_file.close()
	print "Wrote pickle to %s" % filepath
	
def normalise(arr):
	"""
	Normalise between 0 and 1 the data, see `func:unnormalise` for the inverse function
	
	:param arr: the data to normalise
	
	:returns: normed array, lowest value, highest value
	"""
	low = np.amin(arr)
	high = np.amax(arr)
	normed_data = (arr - low) / (high - low)
	return normed_data, low, high

def unnormalise(arr, low, high):
	return arr * (high - low) + low

def compute_rmsd(model, truth):
	"""
	Compute the root-mean-square deviation for the model and the truth.
	
	:param model:
	:param truth:
	"""	
	
	rmsd = np.sqrt(np.mean((model-truth)*(model-truth),axis=1))
	return rmsd

def compute_pca(data, n_components=None):
	"""
	Get the PCA decomposition according to:
	http://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html
	"""
	from sklearn.decomposition import PCA
	
	pca = PCA(n_components=n_components)
	pca.fit(data)
	
	return pca

def mad(nparray):
	"""
	The Median Absolute Deviation
	http://en.wikipedia.org/wiki/Median_absolute_deviation

	Multiply this by 1.4826 to convert into an estimate of the Gaussian std.
	"""

	return np.median(np.fabs(nparray - np.median(nparray)))


def skystats(stamp):
	"""
	I measure some statistics of the pixels along the edge of an image or stamp.
	Useful to measure the sky noise, but also to check for problems. Use "mad"
	directly as a robust estimate the sky std.

	:param stamp: a galsim image, usually a stamp

	:returns: a dict containing "std", "mad", "mean" and "med"
	
	Note that "mad" is already rescaled by 1.4826 to be comparable with std.
	"""

	a = stamp
	edgepixels = np.concatenate([
			a[0,1:], # left
			a[-1,1:], # right
			a[:,0], # bottom
			a[1:-1,-1] # top
			])
	assert len(edgepixels) == 2*(a.shape[0]-1) + 2*(a.shape[0]-1)


	# And we convert the mad into an estimate of the Gaussian std:
	return {
		"std":np.std(edgepixels), "mad": 1.4826 * mad(edgepixels),
		"mean":np.mean(edgepixels), "med":np.median(edgepixels)
		}
	
	
def sigmoid(x):
	return 1. / (1. + np.exp(-x))


def sigmoid_prime(x):
	return sigmoid(x) * (1. - sigmoid(x))


def KL_divergence(x, y):
	return x * np.log(x / y) + (1. - x) * np.log((1. - x) / (1. - y))