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
