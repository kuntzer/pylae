import numpy as np

def corrupt(ae, data, corruption):
	
	if type(corruption) == float:
		cdata = np.random.binomial(size=data.shape, n=1, p=1.-corruption) * data
	else:
		if ae.layers[0].data_std is not None and ae.layers[0].data_norm is not None:
			scales = np.random.uniform(low=corruption[0], high=corruption[1], size=data.shape[1])
			
			data = unnormalise(data, ae.layers[0].data_norm[0], ae.layers[0].data_norm[1])
			data = unstandardize(data, ae.layers[0].data_std[0], ae.layers[0].data_std[1])
			
			p = np.random.binomial
			noise_maps = [np.random.normal(scale=sig, size=data.shape[0]) for sig in scales] #* p(1, 0.5) 
			noise_maps = np.asarray(noise_maps)
			cdata = data + noise_maps.T
			
			cdata, _, _ = standardize(cdata, ae.layers[0].data_std[0], ae.layers[0].data_std[1])
			cdata, _, _ = normalise(cdata, ae.layers[0].data_norm[0], ae.layers[0].data_norm[1])
			
			# Just making sure we're not out of bounds:
			min_thr = 1e-6
			max_thr = 0.99999
			
			#if ((cdata < min_thr).sum() > 0 or (cdata > max_thr).sum() > 0) and False:
			#	print np.amin(data), np.amax(data), np.mean(data), np.std(data)
			#	print 'N/C:', (cdata < min_thr).sum(), (cdata > max_thr).sum()
			#	print np.amin(cdata), np.amax(cdata), np.mean(cdata), np.std(cdata)
			#	print 
			cdata = np.clip(cdata, min_thr, max_thr)
			
	return cdata

def normalise(arr, low=None, high=None):
	"""
	Normalise between 0 and 1 the data, see `func:unnormalise` for the inverse function
	
	:param arr: the data to normalise
	:param low: lowest value of the array, if `None`, computed on `arr`
	:param high: highest value of the array, if `None`, computed on `arr`
	
	:returns: normed array, lowest value, highest value
	"""
	if low is None:
		low = np.amin(arr)
		
	if high is None:
		high = np.amax(arr)
	
	normed_data = (arr - low) / (high - low)
	return normed_data, low, high

def unnormalise(arr, low, high):
	return arr * (high - low) + low

def standardize(data, mean=None, std=None):
	if std is None:
		std = np.std(data)
	if mean is None:
		mean = np.mean(data)
		
	return (data - mean) / std, mean, std

def unstandardize(data, mean, std):
	
	return data * std + mean
