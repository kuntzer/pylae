import os
import numpy as np
import pylae

def load_images(datadir, n):
	"""
	Loads the data and render it black and white
	:param datadir: the directory where the data are
	:param n: number of images to extract
	"""
	
	collected_data = None
	
	for ii in range(1, 6):
		try:
			imgs = pylae.utils.readpickle(os.path.join(datadir, "data_batch_{}".format(ii)))
		except:
			raise IOError("""Files not found -- To use the cifar-10 demos, please download the CIFAR-10 python version (163 MB) file from 
http://www.cs.toronto.edu/~kriz/cifar.html and decompress it in the demo folder. """)
		imgs = imgs["data"].reshape(10000, 1024, 3, order="F")
		imgs = imgs[:,:,0]*0.299 + imgs[:,:,1]*0.587 + imgs[:,:,2]*0.114
		
		if collected_data is None:
			collected_data = imgs 
		else:
			collected_data = np.vstack([collected_data, imgs])

		if np.shape(collected_data)[0] >= n:
			break

	return collected_data[:n] / 255.