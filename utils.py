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
	res[x <= 0.] = alpha

	return res

def relu_prime(x):
	
	res = np.ones_like(x) * x
	res[x <= 0] = 0.
	res[x > 0] = 1.
	
	return res

def leaky_relu_prime(x, alpha=0.01):
	
	res = np.ones_like(x) * x
	res[x <= 0] = alpha
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

def cross_entropy(target, preds):
	cost = - np.sum(target * np.log(preds) + (1. - target) * np.log(1. - preds), axis=1)
	cost = np.mean(cost)
	
	return cost

def new_epoch(cl):
	cl.mini_batch_ids = np.ones(cl.Ndata)
	
def select_mini_batch(cl):
	
	if not hasattr(cl, 'mini_batch_ids') or np.sum(cl.mini_batch_ids) <= 0:
		new_epoch(cl)
	
		if cl.verbose: 
			print "A new epoch has started"
			
	if np.sum(cl.mini_batch_ids) < cl.mini_batch:
		b = int(cl.mini_batch_ids.sum())
	else:
		b = cl.mini_batch

	aids = np.where(cl.mini_batch_ids == 1)[0]
	avail_ids = np.arange(cl.Ndata)[aids]
	ids_batch = np.random.choice(avail_ids, b, replace=False)
	
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

def tile_raster_images(X, img_shape, tile_shape, tile_spacing=(0, 0),
					   scale_rows_to_unit_interval=True,
					   output_pixel_vals=True):
	"""
	Transform an array with one flattened image per row, into an array in
	which images are reshaped and layed out like tiles on a floor.

	This function is useful for visualizing datasets whose rows are images,
	and also columns of matrices for transforming those rows
	(such as the first layer of a neural net).

	:type X: a 2-D ndarray or a tuple of 4 channels, elements of which can
	be 2-D ndarrays or None;
	:param X: a 2-D array in which every row is a flattened image.

	:type img_shape: tuple; (height, width)
	:param img_shape: the original shape of each image

	:type tile_shape: tuple; (rows, cols)
	:param tile_shape: the number of images to tile (rows, cols)

	:param output_pixel_vals: if output should be pixel values (i.e. int8
	values) or floats

	:param scale_rows_to_unit_interval: if the values need to be scaled before
	being plotted to [0,1] or not


	:returns: array suitable for viewing as an image.
	(See:`Image.fromarray`.)
	:rtype: a 2-d array with same dtype as X.

	"""

	assert len(img_shape) == 2
	assert len(tile_shape) == 2
	assert len(tile_spacing) == 2

	# The expression below can be re-written in a more C style as
	# follows :
	#
	# out_shape	= [0,0]
	# out_shape[0] = (img_shape[0]+tile_spacing[0])*tile_shape[0] -
	#				tile_spacing[0]
	# out_shape[1] = (img_shape[1]+tile_spacing[1])*tile_shape[1] -
	#				tile_spacing[1]
	out_shape = [
		(ishp + tsp) * tshp - tsp
		for ishp, tshp, tsp in zip(img_shape, tile_shape, tile_spacing)
	]

	if isinstance(X, tuple):
		assert len(X) == 4
		# Create an output numpy ndarray to store the image
		if output_pixel_vals:
			out_array = np.zeros((out_shape[0], out_shape[1], 4),
									dtype='uint8')
		else:
			out_array = np.zeros((out_shape[0], out_shape[1], 4),
									dtype=X.dtype)

		#colors default to 0, alpha defaults to 1 (opaque)
		if output_pixel_vals:
			channel_defaults = [0, 0, 0, 255]
		else:
			channel_defaults = [0., 0., 0., 1.]

		for i in xrange(4):
			if X[i] is None:
				# if channel is None, fill it with zeros of the correct
				# dtype
				dt = out_array.dtype
				if output_pixel_vals:
					dt = 'uint8'
				out_array[:, :, i] = np.zeros(
					out_shape,
					dtype=dt
				) + channel_defaults[i]
			else:
				# use a recurrent call to compute the channel and store it
				# in the output
				out_array[:, :, i] = tile_raster_images(
					X[i], img_shape, tile_shape, tile_spacing,
					scale_rows_to_unit_interval, output_pixel_vals)
		return out_array

	else:
		# if we are dealing with only one channel
		H, W = img_shape
		Hs, Ws = tile_spacing

		# generate a matrix to store the output
		dt = X.dtype
		if output_pixel_vals:
			dt = 'uint8'
		out_array = np.zeros(out_shape, dtype=dt)

		for tile_row in xrange(tile_shape[0]):
			for tile_col in xrange(tile_shape[1]):
				if tile_row * tile_shape[1] + tile_col < X.shape[0]:
					this_x = X[tile_row * tile_shape[1] + tile_col]
					if scale_rows_to_unit_interval:
						# if we should scale values to be between 0 and 1
						# do this by calling the `scale_to_unit_interval`
						# function
						this_img, _, _ = normalise(
							this_x.reshape(img_shape))
					else:
						this_img = this_x.reshape(img_shape)
					# add the slice to the corresponding position in the
					# output array
					c = 1
					if output_pixel_vals:
						c = 255
					out_array[
						tile_row * (H + Hs): tile_row * (H + Hs) + H,
						tile_col * (W + Ws): tile_col * (W + Ws) + W
					] = this_img * c
		return out_array

def mkdir(somedir):
	"""
	A wrapper around os.makedirs.
	:param somedir: a name or path to a directory which I should make.
	"""
	if not os.path.isdir(somedir):
		os.makedirs(somedir)
		
