import cPickle as pickle 
import gzip
import os
import numpy as np
import logging

logger = logging.getLogger(__name__)

def readpickle(filepath):
	"""
	I read a pickle file and return whatever object it contains.
	If the filepath ends with .gz, I'll unzip the pickle file.
	"""
	pkl_file = open(filepath, 'rb')
	obj = pickle.load(pkl_file)
	pkl_file.close()
	logger.info("Read %s" % filepath)
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
	logger.info("Wrote %s" % filepath)
	

def compute_rmsd(model, truth):
	"""
	Compute the root-mean-square deviation for the model and the truth.
	
	:param model:
	:param truth:
	"""	
	
	rmsd = np.sqrt(np.mean((model-truth)*(model-truth),axis=1))
	return rmsd

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

def relu(x):
	return np.maximum(np.zeros_like(x), x)

def leaky_relu(x, alpha=0.01):
	
	res = np.ones_like(x) * x
	res[x <= 0.] = alpha * x

	return res

def relu_prime(x):
	
	res = np.ones_like(x) * x
	res[x <= 0] = 0.
	res[x > 0] = 1.
	
	return res

def leaky_relu_prime(x, alpha=0.01):
	
	res = np.ones_like(x) * x
	res[x <= 0] = alpha * x
	res[x > 0] = 1.
	
	return res

def KL_divergence(x, y):
	# Making sure that the KL doesn't blow up 
	x = np.clip(x, 1e-6, 0.99999)
	y = np.clip(y, 1e-6, 0.99999)
	return x * np.log(x / y) + (1. - x) * np.log((1. - x) / (1. - y))

def KL_prime(x, y):
	# Making sure that the KL doesn't blow up 
	x = np.clip(x, 1e-6, 0.99999)
	y = np.clip(y, 1e-6, 0.99999)
	return -x/y + (1. - x) / (1. - y)

def cross_entropy(targets, preds):
	targets = np.clip(targets, 1e-6, 0.99999)
	preds = np.clip(preds, 1e-6, 0.99999)
	cost = - np.sum(targets * np.log(preds) + (1. - targets) * np.log(1. - preds), axis=1)
	cost = np.mean(cost)
	
	return cost

def new_epoch(cl):
	cl.mini_batch_ids = np.ones(cl.Ndata)
	
def select_mini_batch(cl):
	
	if not hasattr(cl, 'mini_batch_ids') or np.sum(cl.mini_batch_ids) <= 0:
		new_epoch(cl)
			
	if np.sum(cl.mini_batch_ids) < cl.mini_batch:
		b = int(cl.mini_batch_ids.sum())
	else:
		b = cl.mini_batch

	aids = np.where(cl.mini_batch_ids == 1)[0]
	avail_ids = np.arange(cl.Ndata)[aids]
	ids_batch = avail_ids[:b]#np.random.choice(avail_ids, b, replace=False)
	
	cl.mini_batch_ids[ids_batch] = 0

	return np.arange(cl.Ndata)[ids_batch]

def rmsd(x, y):
	"""
	Returns the RMSD between two numpy arrays
	(only beginners tend to inexactly call this the RMS... ;-)
	
	http://en.wikipedia.org/wiki/Root-mean-square_deviation
	
	This function also works as expected on masked arrays.	
	"""
	return np.sqrt(np.nanmean((x - y)**2.0))

def mkdir(somedir):
	"""
	A wrapper around os.makedirs.
	:param somedir: a name or path to a directory which I should make.
	"""
	if not os.path.isdir(somedir):
		os.makedirs(somedir)
		
