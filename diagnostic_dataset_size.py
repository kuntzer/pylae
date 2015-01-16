import pylae
import pylae.utils as utils

import numpy as np
import pylab as plt
import os		
		
import multiprocessing

network_name = "gauss_psf_"

# Load the data and remember the size of the data (assume a square image here)
# ! The data *must* be normalised here
#dataset = np.loadtxt("data/psfs-gs_rnd.dat", delimiter=",")
dataset = np.loadtxt("data/psfs-smalldev.dat", delimiter=",")

dataset, low, high = utils.normalise(dataset)
size = np.sqrt(np.shape(dataset)[1])

# Can we skip some part of the training ?
do_train = True

# Definition of the first half of the autoencoder -- the encoding bit.
# The deeper the architecture the more complex features can be learned.
architecture = [256, 16]
#architecture = [256, 128, 64, 32]
# The layers_type must have len(architecture)+1 item.
# TODO: explain why and how to choose.
layers_type = ["SIGMOID", "SIGMOID", "LINEAR"]
lock = multiprocessing.Lock()

def worker(datas):
	
	datasizes = []
	rmsd_train = []
	rmsd_test = []

	# Separate into training and testing data
	datasize = np.shape(dataset)[0]
	datasize *= datas
	trainper = 0.7
	ind = np.int(trainper * datasize)

	train_data = dataset[0:ind]
	test_data = dataset[ind:datasize]
	
	print 'Shape of the training set: ', np.shape(train_data)
	print 'Shape of the testing set: ', np.shape(test_data)
	
	datasizes.append(datasize)
	ae = pylae.autoencoder.AutoEncoder(network_name)
	ae.pre_train(train_data, architecture, layers_type, learn_rate={'SIGMOID':0.1, 'LINEAR':1e-2}, 
				iterations=2000, mini_batch=100)

	print 'Starting backpropagation'
	ae.backpropagation(train_data, iterations=500, learn_rate=0.1, momentum_rate=0.85)

	os.system("/usr/bin/canberra-gtk-play --id='complete-media-burn'")

	reconstruc = ae.feedforward(train_data)
	reconstructest = ae.feedforward(test_data)

	# Compute the error on the ellipticity:
	train_ell = []
	recon_train_ell = []
	test_ell = []
	recon_test_ell = []
	for ii in range(ind):
		img = train_data[ii].reshape(size,size)
		g1, g2 = utils.get_ell(img)
		train_ell.append([g1, g2, np.sqrt(g1*g1 + g2*g2)])
		
		img = reconstruc[ii].reshape(size,size)
		g1, g2 = utils.get_ell(img)
		recon_train_ell.append([g1, g2, np.sqrt(g1*g1 + g2*g2)])
		
		if ii >= datasize - ind: continue
			
		img = train_data[ii].reshape(size,size)
		g1, g2 = utils.get_ell(img)
		test_ell.append([g1, g2, np.sqrt(g1*g1 + g2*g2)])
		
		img = reconstructest[ii].reshape(size,size)
		g1, g2 = utils.get_ell(img)
		recon_test_ell.append([g1, g2, np.sqrt(g1*g1 + g2*g2)])
		
	train_ell = np.asarray(train_ell)
	recon_train_ell = np.asarray(recon_train_ell)
	test_ell = np.asarray(test_ell)
	recon_test_ell = np.asarray(recon_test_ell)
	
	ae_ell_error = recon_train_ell[:,2] - train_ell[:,2]
	ae_error_train = [np.sqrt(np.mean(ae_ell_error * ae_ell_error))]
	
	ae_ell_error = recon_test_ell[:,2] - test_ell[:,2]
	ae_error_test = [np.sqrt(np.mean(ae_ell_error * ae_ell_error))]

	#lock.acquire()
	# Compute the RMSD error for the training set
	rmsd_train.append(np.mean(utils.compute_rmsd(train_data, reconstruc)))

	# Compute the RMSD error for the test set
	rmsd_test.append(np.mean(utils.compute_rmsd(test_data, reconstructest)))
	#lock.release()

	return datasizes, rmsd_train, rmsd_test, ae_error_train, ae_error_test

if do_train:
	
	datasizes = []
	rmsd_train = []
	rmsd_test = []
	error_train = []
	error_test = []
	
	datass = [1., 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.15, 0.1]#np.linspace(1,0,10, endpoint=False)
	ncpu=7#multiprocessing.cpu_count()
	#res=worker(0.5)
	pool = multiprocessing.Pool(processes=ncpu)

	res = pool.map(worker, datass)
	res = np.asarray(res)
	for ii in range(len(datass)):
		print res[ii,1]
		datasizes.extend(res[ii, 0])
		rmsd_train.extend(res[ii, 1])
		rmsd_test.extend(res[ii, 2])
		error_train.extend(res[ii, 3])
		error_test.extend(res[ii, 4])

	pool.close()
	pool.join()

	utils.writepickle([datasizes, rmsd_train, rmsd_test, error_train, error_test], "%sdiagnostic_dataset_size.pkl" % network_name)
else:
	datasizes, rmsd_train, rmsd_test, error_train, error_test = utils.readpickle("%sdiagnostic_dataset_size.pkl" % network_name)

print datasizes
print rmsd_test

plt.figure()
pylae.plots.lc_dataset_size(datasizes, error_train, error_test)
plt.show()