# Partly inspired from https://github.com/jatinshah/ufldl_tutorial

import numpy as np
import urllib
import os
import gzip

def load_MNIST_images(filename, dlifneedbe=True):
	"""
	returns a 28x28x[number of MNIST images] matrix containing
	the raw MNIST images
	:param filename: input data file
	"""
	
	if not os.path.exists(filename):
		if dlifneedbe:
			print 'Downloading MNIST data, this might take some time...'
			download_MNIST_data()
			print 'Download completed.'
		else:
			raise IOError("{} not found, cannot load images".format(filename))
	
	with open(filename, "r") as f:
		magic = np.fromfile(f, dtype=np.dtype('>i4'), count=1)

		num_images = np.fromfile(f, dtype=np.dtype('>i4'), count=1)[0]
		num_rows = np.fromfile(f, dtype=np.dtype('>i4'), count=1)[0]
		num_cols = np.fromfile(f, dtype=np.dtype('>i4'), count=1)[0]

		images = np.fromfile(f, dtype=np.ubyte)
		images = images.reshape((num_images, num_rows * num_cols)).transpose()
		images = images.astype(np.float64) / 255

		f.close()

		return images

def load_MNIST_labels(filename):
	"""
	returns a [number of MNIST images]x1 matrix containing
	the labels for the MNIST images

	:param filename: input file with labels
	"""
	with open(filename, 'r') as f:
		magic = np.fromfile(f, dtype=np.dtype('>i4'), count=1)

		num_labels = np.fromfile(f, dtype=np.dtype('>i4'), count=1)

		labels = np.fromfile(f, dtype=np.ubyte)

		f.close()

		return labels

def download_MNIST_data(outdir="mnist-data"):
	"""
	Downloading the MNIST data and storing it to `outdir`.
	"""
	
	url = 'http://yann.lecun.com/exdb/mnist/'
	ifiles = ["train-images-idx3-ubyte.gz", "train-labels-idx1-ubyte.gz", "t10k-images-idx3-ubyte.gz", "t10k-labels-idx1-ubyte.gz"]
	ofiles = ["train-images-idx3-ubyte", "train-labels-idx1-ubyte", "t10k-images-idx3-ubyte", "t10k-labels-idx1-ubyte"]
	
	if not os.path.exists(outdir):
		os.mkdir(outdir)
	
	for ifname, ofname in zip(ifiles, ofiles):
		print 'Downloading {}...'.format(ifname)
		dwnl = urllib.URLopener()
		dwnl.retrieve(os.path.join(url, ifname), os.path.join(outdir, ifname))
		
		print 'Decompressing {}...'.format(ifname)
		compressed = gzip.GzipFile(os.path.join(outdir, ifname), 'rb')
		s = compressed.read()
		compressed.close()

		os.remove(os.path.join(outdir, ifname))
		outf = file(os.path.join(outdir, ofname), 'wb')
		outf.write(s)
		outf.close()
